"""Settings for main.py. Edit the defaults below, then run the script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Choose the input, camera settings, and outputs for a stereo run.

    Paths are relative to the repository unless they are absolute. Disparity
    settings are in pixels. Depth and point-cloud coordinates are in metres.
    """

    workflow: str = "middlebury"  # "middlebury", "calibrate", or "pair"
    output_dir: Path = Path("outputs")
    visualize: bool = True  # Show and save summary plots.

    # Middlebury images are already rectified and come with metric calibration.
    # main.py processes every calibrated scene in this split.
    scene_dir: Path = Path("data/middlebury/MiddEval3/trainingQ")
    evaluation: str = "off"  # "off", "all", or "non_occluded" for Middlebury.

    # Calibration estimates intrinsics and the relative pose from chessboard views.
    chessboard_left_dir: Path = Path("data/calibration/left")
    chessboard_right_dir: Path = Path("data/calibration/right")
    board_size: tuple[int, int] = (9, 6)  # Inner corners, columns then rows.
    square_size_mm: float | None = None  # Measure the real edge for metric depth.
    calibration_path: Path = Path("outputs/calibration.json")

    # Raw stereo images must be rectified with the saved model before matching.
    left_image: Path = Path("data/calibration/left/left06.jpg")
    right_image: Path = Path("data/calibration/right/right06.jpg")
    rectification_alpha: float = 0.0  # 0 crops invalid borders; 1 retains the view.

    # SGBM searches this many candidate shifts in multiples of 16. None uses
    # the Middlebury scene recommendation or the raw-pair fallback.
    num_disparities: int | None = None
    pair_num_disparities: int = 144
    min_disparity: int = 0
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1

    # Limit exported depth and point clouds. Evaluation uses the full estimate.
    max_depth_m: float | None = None
    cloud: bool = True
    cloud_stride: int = 2
    bad_px: float = 2.0
    bad_depth_m: float = 0.1
