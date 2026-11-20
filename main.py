"""Run the stereo vision workflow selected in config.py.

Run this file in PyCharm or with python main.py. Middlebury is the default
because its images are already aligned and its calibration is supplied.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2

from config import Config
from stereo.calibration import (
    StereoCalibration,
    calibrate,
    chessboard_preview,
    find_pairs,
    load_calibration,
    load_middlebury_scene,
    rectify_pair,
    save_calibration,
)
from stereo.reconstruction import reconstruct
from utils import read_disparity_pfm, read_image, read_pair, show_calibration_summary

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
WORKFLOW_ERRORS = (OSError, ValueError, cv2.error, ImportError, RuntimeError)


def _run_directory(config: Config, name: str) -> Path:
    """Build a fresh output path for a scene or raw pair."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return ROOT / config.output_dir / f"{name}-{timestamp}"


def _run_calibration(config: Config) -> None:
    """Fit a stereo model from synchronized chessboards and save it."""
    left_dir = ROOT / config.chessboard_left_dir
    right_dir = ROOT / config.chessboard_right_dir
    LOGGER.info("Finding chessboard pairs in %s and %s", left_dir, right_dir)
    pairs = find_pairs(left_dir, right_dir)
    path = ROOT / config.calibration_path
    if path.exists():
        raise FileExistsError(
            f"Choose an unused calibration_path in config.py: {path}"
        )
    LOGGER.info("Calibrating from %d chessboard pairs", len(pairs))
    model = calibrate(pairs, config.board_size, config.square_size_mm)
    save_calibration(path, model)
    LOGGER.info(
        "Saved %s using %d/%d pairs (stereo RMS %.3f px)",
        path,
        model.pairs_used,
        model.pairs_total,
        model.stereo_rms_px,
    )
    if config.visualize:
        LOGGER.info("Generating calibration summary")
        try:
            _show_calibration(config, pairs, model, path)
        except Exception as error:
            LOGGER.warning("Calibration saved, but its summary failed: %s", error)


def _show_calibration(
    config: Config,
    pairs: list[tuple[Path, Path]],
    model: StereoCalibration,
    path: Path,
) -> None:
    """Render the optional summary for a saved camera model."""
    left, right = chessboard_preview(pairs, config.board_size)
    width, height = model.image_size
    baseline = sum(component**2 for component in model.translation) ** 0.5
    unit = "m" if model.square_size_mm is not None else "board squares"
    metadata = (
        f"Image size   {width} × {height} px\n"
        f"Usable pairs   {model.pairs_used}/{model.pairs_total}\n"
        f"Left reprojection RMS   {model.left_rms_px:.3f} px\n"
        f"Right reprojection RMS   {model.right_rms_px:.3f} px\n"
        f"Stereo reprojection RMS   {model.stereo_rms_px:.3f} px\n"
        f"Baseline   {baseline:.4f} {unit}"
    )
    show_calibration_summary(
        path.with_name(f"{path.stem}_summary.png"), left, right, metadata
    )


def _run_middlebury_scene(config: Config, scene: Path) -> None:
    """Reconstruct one rectified scene using its supplied camera parameters."""
    LOGGER.info("Loading rectified images and calibration for %s", scene.name)
    left, right, geometry, ndisp = load_middlebury_scene(scene)
    # OpenCV needs a multiple of 16. Round up to cover the scene's full range.
    span = ((ndisp + 15) // 16) * 16
    sources = {"scene": scene, "calibration": scene / "calib.txt"}

    reference = None
    reference_mask = None
    if config.evaluation != "off":
        LOGGER.info(
            "Loading %s evaluation references for %s", config.evaluation, scene.name
        )
        reference_path = scene / "disp0GT.pfm"
        mask_path = scene / "mask0nocc.png"
        try:
            disparity = read_disparity_pfm(reference_path)
            if disparity.shape != left.shape[:2]:
                raise ValueError(f"Reference dimensions do not match {scene / 'im0.png'}")
            if config.evaluation == "non_occluded":
                mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask.shape != disparity.shape:
                    raise ValueError(f"Mask dimensions do not match {reference_path}")
                reference_mask = mask == 255
                sources["ground_truth_mask"] = mask_path
            reference = disparity
            sources["ground_truth_disparity"] = reference_path
        except FileNotFoundError as error:
            LOGGER.warning(
                "Skipping evaluation for %s. Missing %s. "
                "Run python download_datasets.py --only ground-truth.",
                scene.name,
                error.filename,
            )
        except (OSError, ValueError, cv2.error) as error:
            LOGGER.warning(
                "Skipping evaluation for %s. Invalid reference: %s", scene.name, error
            )

    reconstruct(
        _run_directory(config, scene.name),
        left,
        right,
        config,
        span if config.num_disparities is None else config.num_disparities,
        geometry=geometry,
        reference=reference,
        reference_mask=reference_mask,
        source_paths=sources,
    )


def _run_middlebury(config: Config) -> None:
    """Reconstruct every scene in the configured Middlebury split in order."""
    if config.evaluation not in ("off", "all", "non_occluded"):
        raise ValueError("Set evaluation in config.py to off, all, or non_occluded")
    split_dir = ROOT / config.scene_dir
    LOGGER.info(
        "Scanning Middlebury split %s (evaluation: %s)", split_dir, config.evaluation
    )
    scenes = sorted(path for path in split_dir.glob("*") if path.is_dir())
    if not scenes:
        raise FileNotFoundError(
            f"No scene folders found in Middlebury split {split_dir}. "
            "Set scene_dir to a split directory such as trainingQ. "
            "To download the dataset, run python download_datasets.py --only middlebury."
        )

    failed = []
    for index, scene in enumerate(scenes, start=1):
        LOGGER.info("Scene %d of %d: %s", index, len(scenes), scene.name)
        try:
            _run_middlebury_scene(config, scene)
        except WORKFLOW_ERRORS as error:
            failed.append(scene.name)
            LOGGER.error("Scene %s failed: %s", scene.name, error)
    completed = len(scenes) - len(failed)
    if failed:
        raise RuntimeError(
            f"Reconstructed {completed}/{len(scenes)} scenes. Failed: {', '.join(failed)}"
        )
    LOGGER.info("Reconstructed all %d scenes", completed)


def _run_raw_pair(config: Config) -> None:
    """Rectify the configured images with a saved model, then reconstruct them."""
    model_path = ROOT / config.calibration_path
    left_path, right_path = ROOT / config.left_image, ROOT / config.right_image
    LOGGER.info("Loading stereo calibration from %s", model_path)
    model = load_calibration(model_path)
    LOGGER.info("Loading stereo images %s and %s", left_path, right_path)
    raw_left, raw_right = read_pair(left_path, right_path)
    LOGGER.info("Rectifying stereo pair (alpha %.2f)", config.rectification_alpha)
    left, right, rois, geometry = rectify_pair(
        raw_left, raw_right, model, config.rectification_alpha
    )
    num_disparities = config.num_disparities
    if num_disparities is None:
        num_disparities = config.pair_num_disparities
    reconstruct(
        _run_directory(config, "pair"),
        left,
        right,
        config,
        num_disparities,
        geometry=geometry,
        rois=rois,
        source_paths={"left": left_path, "right": right_path, "calibration": model_path},
        raw_inputs=(raw_left, raw_right) if config.visualize else None,
    )


def main(config: Config | None = None) -> None:
    """Run the workflow selected in config.py.

    Args:
        config: Optional settings to use in place of the defaults in config.py.

    Raises:
        OSError: An input is missing or an output cannot be written.
        ValueError: The workflow, input data, or camera geometry is invalid.
        RuntimeError: One or more Middlebury scenes failed reconstruction.
    """
    config = Config() if config is None else config
    LOGGER.info(
        "Starting %s workflow (visualization %s)",
        config.workflow,
        "on" if config.visualize else "off",
    )
    if config.workflow == "middlebury":
        _run_middlebury(config)
    elif config.workflow == "calibrate":
        _run_calibration(config)
    elif config.workflow == "pair":
        _run_raw_pair(config)
    else:
        raise ValueError("Set workflow in config.py to middlebury, calibrate, or pair")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except WORKFLOW_ERRORS as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from None
