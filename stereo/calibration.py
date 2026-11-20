"""Calibrate stereo cameras, rectify pairs, and load Middlebury geometry."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from utils import Image, read_pair, read_scene_metadata

LOGGER = logging.getLogger(__name__)
Matrix = NDArray[np.float64]
ROI = Tuple[int, int, int, int]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class StereoCalibration:
    """OpenCV pinhole model and the transform from left to right camera.

    Translation is in metres when square_size_mm is provided and in board
    squares otherwise. Image dimensions are stored as (width, height).
    """

    image_size: tuple[int, int]
    left_matrix: Matrix
    left_distortion: Matrix
    right_matrix: Matrix
    right_distortion: Matrix
    rotation: Matrix
    translation: Matrix
    square_size_mm: float | None
    left_rms_px: float
    right_rms_px: float
    stereo_rms_px: float
    pairs_used: int
    pairs_total: int

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(
            type(value) is not int or value <= 0 for value in self.image_size
        ):
            raise ValueError("Invalid calibration image size")
        for matrix in (self.left_matrix, self.right_matrix):
            if (
                matrix.shape != (3, 3)
                or not np.isfinite(matrix).all()
                or min(matrix[0, 0], matrix[1, 1]) <= 0
                or not np.allclose(np.tril(matrix, -1), 0)
                or not np.isclose(matrix[0, 1], 0)
                or not np.isclose(matrix[2, 2], 1)
            ):
                raise ValueError("Invalid pinhole camera matrix")
        for distortion in (self.left_distortion, self.right_distortion):
            if (
                distortion.shape not in {(n,) for n in (4, 5, 8, 12, 14)}
                or not np.isfinite(distortion).all()
            ):
                raise ValueError("Invalid OpenCV distortion coefficients")
        if (
            self.rotation.shape != (3, 3)
            or not np.isfinite(self.rotation).all()
            or not np.allclose(self.rotation.T @ self.rotation, np.eye(3))
            or not np.isclose(np.linalg.det(self.rotation), 1)
        ):
            raise ValueError("Invalid stereo rotation matrix")
        if (
            self.translation.shape != (3,)
            or not np.isfinite(self.translation).all()
            or not np.isfinite(np.linalg.norm(self.translation))
            or np.linalg.norm(self.translation) == 0
        ):
            raise ValueError("Invalid stereo translation")
        _validate_square_size(self.square_size_mm)
        errors = (self.left_rms_px, self.right_rms_px, self.stereo_rms_px)
        if not np.isfinite(errors).all() or min(errors) < 0:
            raise ValueError("Invalid calibration reprojection errors")
        if not 3 <= self.pairs_used <= self.pairs_total:
            raise ValueError("Invalid calibration pair counts")


def _validate_square_size(square_size_mm: float | None) -> None:
    if square_size_mm is not None and (
        not np.isfinite(square_size_mm) or square_size_mm <= 0
    ):
        raise ValueError("Square size must be positive and finite in millimetres")


def _index_images(directory: Path, side: str) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem.lower()
        # The OpenCV samples use left01/right01. Custom captures may share a stem.
        match = re.fullmatch(rf"{side}[_-]?(\d+)", stem)
        key = match[1] if match else stem
        if key in images:
            raise ValueError(f"Duplicate calibration image ID {key!r} in {directory}")
        images[key] = path
    return images


def find_pairs(left_dir: Path, right_dir: Path) -> list[tuple[Path, Path]]:
    """Pair images by shared stem or an OpenCV-style side prefix and index."""
    left = _index_images(left_dir, "left")
    right = _index_images(right_dir, "right")
    if not left or left.keys() != right.keys():
        raise ValueError(
            "Calibration image IDs must match across both folders. "
            f"Left only: {sorted(left.keys() - right.keys())}. "
            f"Right only: {sorted(right.keys() - left.keys())}."
        )
    return [(left[key], right[key]) for key in sorted(left)]


def _detect_corners(
    image: Image, board_size: tuple[int, int]
) -> NDArray[np.float32] | None:
    """Locate inner chessboard corners to subpixel precision."""
    found, corners = cv2.findChessboardCornersSB(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        board_size,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    return corners if found else None


def chessboard_preview(
    pairs: list[tuple[Path, Path]], board_size: tuple[int, int]
) -> tuple[Image, Image]:
    """Draw the detected corners on the first usable left and right chessboard pair."""
    for left_path, right_path in pairs:
        left, right = read_pair(left_path, right_path)
        left_corners = _detect_corners(left, board_size)
        right_corners = _detect_corners(right, board_size)
        if left_corners is not None and right_corners is not None:
            cv2.drawChessboardCorners(left, board_size, left_corners, True)
            cv2.drawChessboardCorners(right, board_size, right_corners, True)
            return left, right
    raise ValueError("No detected chessboard pair to visualize")


def calibrate(
    pairs: list[tuple[Path, Path]],
    board_size: tuple[int, int],
    square_size_mm: float | None = None,
) -> StereoCalibration:
    """Estimate intrinsics per camera, then fit the stereo transform.

    Args:
        pairs: Synchronized paths with corresponding left and right views.
        board_size: Number of inner corners as (columns, rows).
        square_size_mm: Measured square edge, or None for a scale-free model.

    Returns:
        Camera matrices, distortion, relative pose and fit diagnostics.
    """
    if min(board_size) < 3 or len(pairs) < 3:
        raise ValueError("Calibration needs at least three pairs and 3x3 inner corners")
    _validate_square_size(square_size_mm)
    columns, rows = board_size
    grid = np.zeros((rows * columns, 3), np.float32)
    grid[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2)
    if square_size_mm is not None:
        grid *= square_size_mm / 1000

    image_size = None
    object_points, left_points, right_points = [], [], []
    LOGGER.info("Detecting chessboard corners")
    for left_path, right_path in pairs:
        left, right = read_pair(left_path, right_path)
        size = (left.shape[1], left.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError(f"Calibration image size changed at {left_path}")
        image_size = size
        left_corners = _detect_corners(left, board_size)
        right_corners = _detect_corners(right, board_size)
        if left_corners is None or right_corners is None:
            LOGGER.warning(
                "Skipping incomplete chessboard pair %s / %s", left_path, right_path
            )
            continue
        object_points.append(grid)
        left_points.append(left_corners)
        right_points.append(right_corners)

    if len(object_points) < 3 or image_size is None:
        raise ValueError("Fewer than three chessboards detected in both cameras")

    LOGGER.info("Fitting camera intrinsics from %d usable pairs", len(object_points))
    left_rms, left_k, left_d, _, _ = cv2.calibrateCamera(
        object_points, left_points, image_size, None, None
    )
    right_rms, right_k, right_d, _, _ = cv2.calibrateCamera(
        object_points, right_points, image_size, None, None
    )
    # Fix each camera's intrinsics so this fit solves the shared relative pose.
    LOGGER.info("Fitting the relative pose of the stereo cameras")
    rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_k,
        left_d,
        right_k,
        right_d,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6),
    )
    return StereoCalibration(
        image_size,
        left_k,
        left_d.ravel(),
        right_k,
        right_d.ravel(),
        rotation,
        translation.ravel(),
        square_size_mm,
        left_rms,
        right_rms,
        rms,
        len(object_points),
        len(pairs),
    )


def save_calibration(path: Path, calibration: StereoCalibration) -> None:
    """Save a versioned JSON model with explicit translation units."""
    record = asdict(calibration)
    for name, value in record.items():
        if isinstance(value, np.ndarray):
            record[name] = value.tolist()
    record["version"] = 1
    record["translation_unit"] = (
        "m" if calibration.square_size_mm is not None else "square"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        # Publish the complete file with exclusive creation.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def load_calibration(path: Path) -> StereoCalibration:
    """Read and validate a model saved by save_calibration."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.pop("version", None) != 1:
            raise ValueError("Expected a version 1 stereo calibration object")
        unit = record.pop("translation_unit")
        expected_unit = "m" if record["square_size_mm"] is not None else "square"
        if unit != expected_unit:
            raise ValueError("Inconsistent calibration translation units")
        if record.keys() != {field.name for field in fields(StereoCalibration)}:
            raise ValueError("Unexpected or missing calibration fields")
        record["image_size"] = tuple(record["image_size"])
        for name in (
            "left_matrix",
            "left_distortion",
            "right_matrix",
            "right_distortion",
            "rotation",
            "translation",
        ):
            record[name] = np.asarray(record[name], dtype=np.float64)
        return StereoCalibration(**record)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid stereo calibration {path}: {error}") from error


@dataclass(frozen=True)
class RectifiedGeometry:
    """Rectified left intrinsics, right-minus-left cx and baseline in metres."""

    fx: float
    fy: float
    cx: float
    cy: float
    disparity_offset_px: float
    baseline_m: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                [
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    self.disparity_offset_px,
                    self.baseline_m,
                ]
            ).all()
            or min(self.fx, self.fy, self.baseline_m) <= 0
        ):
            raise ValueError(
                "Geometry must be finite with positive focal lengths and baseline"
            )

    def reprojection_matrix(self) -> NDArray[np.float64]:
        """Build OpenCV's homogeneous (x, y, disparity, 1) to XYZ transform."""
        # The y factor allows fx != fy while keeping Z = fx * B / (d + offset).
        return np.array(
            [
                [1, 0, 0, -self.cx],
                [0, self.fx / self.fy, 0, -self.cy * self.fx / self.fy],
                [0, 0, 0, self.fx],
                [0, 0, 1 / self.baseline_m, self.disparity_offset_px / self.baseline_m],
            ],
            dtype=np.float64,
        )


def rectify_pair(
    left: Image,
    right: Image,
    calibration: StereoCalibration,
    alpha: float,
) -> tuple[Image, Image, tuple[ROI, ROI], RectifiedGeometry | None]:
    """Rectify synchronized BGR images at their calibration resolution.

    Args:
        left: Original left camera image.
        right: Original right camera image.
        calibration: Model for the same rig, lens settings and resolution.
        alpha: Crop to valid pixels at 0 or retain the full field of view at 1.

    Returns:
        Both remapped images, their valid rectangles and metric geometry if known.
    """
    if left.shape != right.shape or left.shape[1::-1] != calibration.image_size:
        raise ValueError(
            f"Stereo pair must match calibration size {calibration.image_size}"
        )
    if not 0 <= alpha <= 1:
        raise ValueError("Rectification alpha must be between 0 and 1")

    size = calibration.image_size
    # R1/R2 rotate each view into a shared geometry. P1/P2 describe the
    # virtual cameras and provide the focal length and baseline for depth.
    r1, r2, p1, p2, _, roi1, roi2 = cv2.stereoRectify(
        calibration.left_matrix,
        calibration.left_distortion,
        calibration.right_matrix,
        calibration.right_distortion,
        size,
        calibration.rotation,
        calibration.translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    # A right camera to the right of the left has a negative projection shift.
    if not np.isclose(p2[1, 3], 0) or not np.isfinite(p2[0, 3]) or p2[0, 3] >= 0:
        raise ValueError(
            "Expected a horizontal rig with the left and right cameras in order"
        )

    geometry = None
    if calibration.square_size_mm is not None:
        geometry = RectifiedGeometry(
            p1[0, 0],
            p1[1, 1],
            p1[0, 2],
            p1[1, 2],
            p2[0, 2] - p1[0, 2],
            -p2[0, 3] / p2[0, 0],
        )
    left_maps = cv2.initUndistortRectifyMap(
        calibration.left_matrix, calibration.left_distortion, r1, p1, size, cv2.CV_16SC2
    )
    right_maps = cv2.initUndistortRectifyMap(
        calibration.right_matrix,
        calibration.right_distortion,
        r2,
        p2,
        size,
        cv2.CV_16SC2,
    )
    return (
        cv2.remap(left, *left_maps, cv2.INTER_LINEAR),
        cv2.remap(right, *right_maps, cv2.INTER_LINEAR),
        (roi1, roi2),
        geometry,
    )


def load_middlebury_scene(scene: Path) -> tuple[Image, Image, RectifiedGeometry, int]:
    """Read both images, their metric geometry and the recommended search span.

    Eval3 supplies separate metadata for each resolution. Validate the images
    against that metadata so a mismatched focal length cannot silently scale depth.
    """
    path = scene / "calib.txt"
    try:
        values = read_scene_metadata(path)
        left_k, right_k = (
            np.asarray(
                [
                    [float(item) for item in row.split()]
                    for row in values[key].strip("[]").split(";")
                ]
            )
            for key in ("cam0", "cam1")
        )
        for camera in (left_k, right_k):
            if (
                camera.shape != (3, 3)
                or not np.isfinite(camera).all()
                or min(camera[0, 0], camera[1, 1]) <= 0
                or not np.allclose(camera[2], [0, 0, 1])
                or not np.allclose([camera[0, 1], camera[1, 0]], 0)
            ):
                raise ValueError("Invalid rectified camera matrix")
        if not np.allclose(left_k[[0, 1, 1], [0, 1, 2]], right_k[[0, 1, 1], [0, 1, 2]]):
            raise ValueError(
                "Expected common focal lengths and vertical principal point"
            )
        offset = float(values["doffs"])
        if not np.isclose(offset, right_k[0, 2] - left_k[0, 2], rtol=0, atol=0.002):
            raise ValueError("Disparity offset disagrees with camera principal points")
        geometry = RectifiedGeometry(
            left_k[0, 0],
            left_k[1, 1],
            left_k[0, 2],
            left_k[1, 2],
            offset,
            float(values["baseline"]) / 1000,
        )
        size = (int(values["width"]), int(values["height"]))
        ndisp = int(values["ndisp"])
        if min(*size, ndisp) <= 0:
            raise ValueError("Image dimensions and disparity span must be positive")
    except (KeyError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid Middlebury metadata {path}: {error}") from error

    left, right = read_pair(scene / "im0.png", scene / "im1.png")
    if left.shape[1::-1] != size:
        raise ValueError(f"Middlebury images do not match calibration size {size}")
    return left, right, geometry, ndisp
