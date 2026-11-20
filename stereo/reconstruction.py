"""Estimate disparity and turn calibrated matches into metric depth and 3D."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from numpy.typing import NDArray

from config import Config
from utils import (
    FloatMap,
    Image,
    colorize,
    epipolar_overlay,
    show_reconstruction_summary,
    write_png,
)

from .calibration import ROI, RectifiedGeometry

LOGGER = logging.getLogger(__name__)


def _decode_disparity(raw: NDArray[np.int16], min_disparity: int) -> FloatMap:
    """Decode 4 fractional bits and mask OpenCV's search-range sentinel."""
    disparity = raw.astype(np.float32) / 16
    disparity[raw <= (min_disparity - 1) * 16] = np.nan
    return disparity


def estimate_disparity(
    left: Image,
    right: Image,
    config: Config,
    num_disparities: int,
    rois: tuple[ROI, ROI] | None = None,
) -> FloatMap:
    """Estimate left-view disparity and mask pixels outside the valid search area.

    Args:
        left: Rectified left BGR image.
        right: Rectified right BGR image with the same dimensions.
        config: StereoSGBM settings.
        num_disparities: Number of horizontal shifts to test, in multiples of 16.
        rois: Optional valid left and right rectangles returned by stereoRectify.

    Returns:
        A float32 disparity map in pixels, with NaN for invalid matches.
    """
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ValueError("StereoSGBM expects matching BGR images")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("StereoSGBM expects uint8 images")

    if num_disparities <= 0 or num_disparities % 16:
        raise ValueError("Search span must be a positive multiple of 16")
    # OpenCV stores disparity and its invalid sentinel in signed 16-bit values.
    if (
        not -2047
        <= config.min_disparity
        < config.min_disparity + num_disparities
        <= 2048
    ):
        raise ValueError("Search range exceeds OpenCV's fixed-point disparity limits")
    if not 1 <= config.block_size <= 31 or config.block_size % 2 == 0:
        raise ValueError("Block size must be odd and between 1 and 31")
    if not 0 <= config.uniqueness_ratio <= 100:
        raise ValueError("Uniqueness ratio must be between 0 and 100")
    if (
        min(config.speckle_window_size, config.speckle_range) < 0
        or config.disp12_max_diff < 1
    ):
        raise ValueError(
            "Speckle settings must be nonnegative and consistency tolerance positive"
        )

    height, width = left.shape[:2]
    full = (0, 0, width, height)
    x, y, roi_width, roi_height = cv2.getValidDisparityROI(
        *(rois or (full, full)),
        config.min_disparity,
        num_disparities,
        config.block_size,
    )
    # OpenCV's ROI does not shrink the right boundary for negative min disparity.
    right_roi = full if rois is None else rois[1]
    end_x = min(
        x + roi_width,
        right_roi[0] + right_roi[2] + config.min_disparity - config.block_size // 2,
    )
    if end_x <= x or roi_height <= 0:
        raise ValueError(
            "No valid matching area. Reduce the disparity range or block size."
        )

    # P1 penalizes a one-pixel disparity change and P2 penalizes larger jumps.
    block_area = config.block_size**2
    matcher = cv2.StereoSGBM_create(
        minDisparity=config.min_disparity,
        numDisparities=num_disparities,
        blockSize=config.block_size,
        P1=8 * block_area,
        P2=32 * block_area,
        uniquenessRatio=config.uniqueness_ratio,
        speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range,
        disp12MaxDiff=config.disp12_max_diff,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = matcher.compute(
        cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
    )
    result = np.full((height, width), np.nan, dtype=np.float32)
    region = np.s_[y : y + roi_height, x:end_x]
    result[region] = _decode_disparity(raw[region], config.min_disparity)
    return result


def depth_from_disparity(disparity: FloatMap, geometry: RectifiedGeometry) -> FloatMap:
    """Convert left-view disparity in pixels to axial depth in metres.

    Args:
        disparity: Left-minus-right pixel coordinates, with NaN for invalid matches.
        geometry: Metric calibration for the same rectified images.

    Returns:
        A float32 depth map with NaN for invalid or non-positive effective disparity.
    """
    if disparity.ndim != 2:
        raise ValueError("Disparity must be a 2D map")
    denominator = disparity.astype(np.float64) + geometry.disparity_offset_px
    valid = np.isfinite(denominator) & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore"):
        np.divide(
            geometry.fx * geometry.baseline_m, denominator, out=depth, where=valid
        )
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


def save_cloud(
    path: Path,
    disparity: FloatMap,
    depth: FloatMap,
    left: Image,
    geometry: RectifiedGeometry,
    stride: int,
) -> int:
    """Write PLY XYZ in metres and left-view RGB, returning the vertex count.

    Args:
        path: Destination PLY path.
        disparity: Left-view disparity in pixels.
        depth: Metric depth, with any rejected points marked NaN.
        left: Rectified left BGR image.
        geometry: The same calibration used to compute depth.
        stride: Retain at most one point every stride pixels in each direction.

    Returns:
        The number of saved vertices. An empty cloud produces no file.
    """
    if stride < 1:
        raise ValueError("Point-cloud stride must be positive")
    if (
        disparity.ndim != 2
        or disparity.shape != depth.shape
        or left.shape != (*depth.shape, 3)
    ):
        raise ValueError("Point cloud requires aligned disparity, depth and color")
    sample = np.s_[::stride, ::stride]
    sampled_depth = depth[sample]
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
    if not valid.any():
        return 0

    # Coordinates in Q refer to the full image. Apply stride after reprojection.
    xyz = cv2.reprojectImageTo3D(disparity, geometry.reprojection_matrix())[sample]
    valid &= np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0)
    if not valid.any():
        return 0

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz[valid].astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(left[sample][valid, ::-1] / 255.0)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f"Open3D could not write {path}")
    return len(cloud.points)


def evaluate_map(
    prediction: FloatMap,
    reference: FloatMap,
    threshold: float,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[dict[str, float | int | None], FloatMap]:
    """Measure errors on finite overlap and count missing estimates as bad.

    Args:
        prediction: Estimated values, with invalid pixels marked NaN.
        reference: Ground truth in the same coordinates and units.
        threshold: Absolute error strictly above which a reference pixel is bad.
        mask: Optional boolean reference region, such as non-occluded pixels.

    Returns:
        Metrics and a float32 absolute error map with NaN outside the overlap.
    """
    if prediction.ndim != 2 or prediction.shape != reference.shape:
        raise ValueError("Prediction and ground truth must be aligned 2D maps")
    if mask is not None and (mask.shape != reference.shape or mask.dtype != np.bool_):
        raise ValueError(
            "Evaluation mask must be boolean and match the reference shape"
        )
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Bad-pixel threshold must be positive and finite")

    eligible = np.isfinite(reference)
    if mask is not None:
        eligible &= mask
    overlap = eligible & np.isfinite(prediction)
    target_count = int(eligible.sum())
    matched_count = int(overlap.sum())
    errors = np.abs(prediction[overlap].astype(np.float64) - reference[overlap])
    bad_count = int((errors > threshold).sum()) + target_count - matched_count
    error_map = np.full(reference.shape, np.nan, dtype=np.float32)
    error_map[overlap] = errors
    metrics = {
        "reference_pixels": target_count,
        "estimated_pixels": matched_count,
        "coverage_pct": 100 * matched_count / target_count if target_count else None,
        "mae": float(errors.mean()) if matched_count else None,
        "rmse": float(np.sqrt(np.mean(errors**2))) if matched_count else None,
        "bad_threshold": threshold,
        "bad_pct_including_missing": 100 * bad_count / target_count
        if target_count
        else None,
    }
    return metrics, error_map


def reconstruct(
    output: Path,
    left: Image,
    right: Image,
    config: Config,
    num_disparities: int,
    geometry: RectifiedGeometry | None = None,
    rois: tuple[ROI, ROI] | None = None,
    reference: FloatMap | None = None,
    reference_mask: NDArray[np.bool_] | None = None,
    *,
    source_paths: dict[str, Path] | None = None,
    raw_inputs: tuple[Image, Image] | None = None,
) -> None:
    """Compute, evaluate and save a rectified stereo reconstruction.

    Args:
        output: New run directory. The result is published after writes succeed.
        left: Rectified left BGR image.
        right: Rectified right BGR image.
        config: Matching, export, and evaluation config.
        num_disparities: Resolved search span for this image pair.
        geometry: Validated metric calibration, when available.
        rois: Optional remapped image regions from stereoRectify.
        reference: Optional rectified left disparity ground truth in pixels.
        reference_mask: Optional boolean region to include in evaluation.
        source_paths: Optional named input paths recorded in the run report.
        raw_inputs: Original views to show when rectification was required.
    """
    if output.exists():
        raise FileExistsError(f"Choose an unused output directory: {output}")
    if geometry is None and config.max_depth_m is not None:
        raise ValueError("A metric calibration is required to set max_depth_m")
    if reference_mask is not None and reference is None:
        raise ValueError("An evaluation mask requires ground-truth disparity")
    if reference is not None and reference.shape != left.shape[:2]:
        raise ValueError(
            "Ground-truth disparity does not match the rectified image dimensions"
        )
    if config.cloud and geometry is not None and config.cloud_stride < 1:
        raise ValueError("Point-cloud stride must be positive")
    if config.max_depth_m is not None and (
        not np.isfinite(config.max_depth_m) or config.max_depth_m <= 0
    ):
        raise ValueError("Maximum depth must be positive and finite")

    # Keep a run together. If a write fails, the temporary directory is cleaned up.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".stereo-", dir=output.parent) as workspace:
        staging = Path(workspace) / "result"
        staging.mkdir()
        LOGGER.info(
            "Matching disparity with StereoSGBM (search %d to %d px, block size %d)",
            config.min_disparity,
            config.min_disparity + num_disparities - 1,
            config.block_size,
        )
        disparity = estimate_disparity(left, right, config, num_disparities, rois)
        report = {
            "image_size": [left.shape[1], left.shape[0]],
            "disparity_unit": "px",
            "disparity_valid_pct": 100 * float(np.isfinite(disparity).mean()),
            "opencv_version": cv2.__version__,
            "settings": {
                name: value.as_posix() if isinstance(value, Path) else value
                for name, value in asdict(config).items()
            },
        }
        report["settings"]["num_disparities"] = num_disparities
        if source_paths is not None:
            report["inputs"] = {
                name: path.as_posix() for name, path in source_paths.items()
            }
        if report["disparity_valid_pct"] == 0:
            LOGGER.warning("StereoSGBM returned no valid matches")
        else:
            LOGGER.info(
                "Disparity matched for %.1f%% of pixels", report["disparity_valid_pct"]
            )
        metrics = {}
        ranges = {}
        depth = None

        def save_map(name: str, values: FloatMap) -> None:
            np.save(staging / f"{name}.npy", values, allow_pickle=False)
            preview, interval = colorize(values)
            write_png(staging / f"{name}.png", preview)
            ranges[name] = interval

        LOGGER.info("Saving disparity and alignment previews")
        save_map("disparity_px", disparity)
        alignment = epipolar_overlay(left, right)
        write_png(staging / "epipolar.png", alignment)
        if rois is not None:
            write_png(staging / "left_rectified.png", left)
            write_png(staging / "right_rectified.png", right)

        if geometry is not None:
            # The same disparity can imply different distances for different
            # rigs, so convert it only when the calibrated baseline is known.
            LOGGER.info("Triangulating metric depth (baseline %.4f m)", geometry.baseline_m)
            unfiltered_depth = depth_from_disparity(disparity, geometry)
            depth = unfiltered_depth
            if config.max_depth_m is not None:
                LOGGER.info("Limiting exported depth to %.3f m", config.max_depth_m)
                depth = np.where(depth > config.max_depth_m, np.nan, depth)
            save_map("depth_m", depth)
            report.update(
                depth_unit="m",
                geometry=asdict(geometry),
                depth_valid_pct=100 * float(np.isfinite(depth).mean()),
            )
            if config.cloud:
                LOGGER.info("Exporting point cloud (stride %d)", config.cloud_stride)
                count = save_cloud(
                    staging / "cloud_m.ply",
                    disparity,
                    depth,
                    left,
                    geometry,
                    config.cloud_stride,
                )
                report["cloud_vertices"] = count
                if count:
                    LOGGER.info("Exported %s point-cloud vertices", f"{count:,}")
                else:
                    LOGGER.warning("No valid points for the PLY")
        else:
            LOGGER.warning(
                "Calibration has no measured square size. Saving disparity only."
            )

        if reference is not None:
            LOGGER.info("Evaluating against ground truth")
            metrics["disparity_px"], error = evaluate_map(
                disparity,
                reference,
                config.bad_px,
                reference_mask,
            )
            save_map("disparity_error_px", error)
            if geometry is not None:
                # Middlebury supplies disparity as the reference. Convert it
                # with the same geometry before comparing depth in metres.
                reference_depth = depth_from_disparity(reference, geometry)
                metrics["depth_m"], error = evaluate_map(
                    unfiltered_depth,
                    reference_depth,
                    config.bad_depth_m,
                    reference_mask,
                )
                save_map("depth_error_m", error)
            report["metrics"] = metrics

        report["preview_ranges"] = ranges
        (staging / "run.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(output.resolve())
    LOGGER.info("Saved reconstruction to %s", output)
    if config.visualize:
        LOGGER.info("Generating summary for %s", output.name)
        try:
            show_reconstruction_summary(
                output, raw_inputs or (left, right), alignment, disparity, depth, report
            )
        except Exception as error:
            LOGGER.warning(
                "Reconstruction saved, but summary failed for %s: %s", output, error
            )
