"""Read stereo inputs and render the resulting image maps."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)
Image = NDArray[np.uint8]
FloatMap = NDArray[np.float32]


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> Image:
    """Decode an image and preserve pixel coordinates from the stored file."""
    # imdecode accepts bytes even when a Windows path contains Unicode.
    data = path.read_bytes()
    image = (
        cv2.imdecode(
            np.frombuffer(data, np.uint8), flags | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if data
        else None
    )
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def read_pair(left_path: Path, right_path: Path) -> tuple[Image, Image]:
    """Read two BGR images with identical dimensions."""
    left, right = read_image(left_path), read_image(right_path)
    if left.shape != right.shape:
        raise ValueError(f"Image dimensions differ: {left_path} and {right_path}")
    return left, right


def write_png(path: Path, image: Image) -> None:
    """Encode and save a PNG, raising an error if encoding fails."""
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode image: {path}")
    path.write_bytes(data.tobytes())


def read_scene_metadata(path: Path) -> dict[str, str]:
    """Read Middlebury's key-value metadata and reject duplicate keys.

    Args:
        path: Scene's calib.txt file.

    Returns:
        Metadata values as strings for the camera model to validate.
    """
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in values:
            raise ValueError(f"Duplicate calibration key {key!r}")
        values[key] = value
    return values


def read_disparity_pfm(path: Path) -> FloatMap:
    """Read raw pixel disparities from a scalar PFM using Eval3 conventions."""
    try:
        with path.open("rb") as stream:
            if stream.readline().strip() != b"Pf":
                raise ValueError("Expected a single-channel PFM (Pf)")
            width, height = map(int, stream.readline().split())
            scale = float(stream.readline())
            if min(width, height) <= 0 or not np.isfinite(scale) or scale == 0:
                raise ValueError("Invalid dimensions or scale")
            data = stream.read()
        if len(data) != width * height * 4:
            raise ValueError("Payload length does not match PFM dimensions")
        # Eval3 uses only the sign for byte order. Its payload already stores
        # pixel disparities even when the header magnitude is 0.003922.
        pixels = np.frombuffer(data, dtype="<f4" if scale < 0 else ">f4")
        return pixels.reshape(height, width)[::-1].astype(np.float32)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid PFM {path}: {error}") from error


def colorize(values: FloatMap) -> tuple[Image, tuple[float, float] | None]:
    """Render finite values with Turbo between the 2nd and 98th percentiles.

    Invalid pixels are black. Return the displayed interval for interpretation.
    """
    valid = np.isfinite(values)
    display = np.zeros(values.shape, np.uint8)
    if not valid.any():
        return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR), None
    samples = values[valid].astype(np.float64)
    lower, upper = (float(value) for value in np.percentile(samples, [2, 98]))
    display[valid] = (
        np.round(255 * np.clip((samples - lower) / (upper - lower), 0, 1))
        if upper > lower
        else 127
    )
    colors = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
    colors[~valid] = 0
    return colors, (lower, upper)


def epipolar_overlay(left: Image, right: Image) -> Image:
    """Join rectified images and draw horizontal guides across both views."""
    combined = np.hstack((left, right))
    step = max(1, combined.shape[0] // 12)
    for row in range(step, combined.shape[0], step):
        cv2.line(combined, (0, row), (combined.shape[1] - 1, row), (0, 220, 0), 1)
    return combined


def _save_and_show(figure: Figure, path: Path) -> None:
    """Save a summary before opening it on an interactive plotting backend."""
    import matplotlib.pyplot as plt

    try:
        figure.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.15)
        LOGGER.info("Saved visualization to %s", path)
        # Agg renders files without a desktop window.
        if plt.get_backend().lower() != "agg":
            LOGGER.info("Close the summary window to continue")
            plt.show(block=True)
    finally:
        plt.close(figure)


def _summary_text(report: dict[str, Any]) -> str:
    """Format the measurements and matching settings recorded in run.json."""
    width, height = report["image_size"]
    settings = report["settings"]
    start = settings["min_disparity"]
    end = start + settings["num_disparities"] - 1
    lines = [
        f"Image size   {width} × {height} px",
        f"Disparity search   {start} to {end} px",
        f"Matching block   {settings['block_size']} px",
        f"Valid disparity   {report['disparity_valid_pct']:.1f}%",
    ]
    if "geometry" in report:
        geometry = report["geometry"]
        lines.extend(
            [
                "",
                f"Valid depth   {report['depth_valid_pct']:.1f}%",
                f"Focal length (fx)   {geometry['fx']:.2f} px",
                f"Baseline   {geometry['baseline_m']:.4f} m",
                f"Disparity offset   {geometry['disparity_offset_px']:.2f} px",
            ]
        )
    if "cloud_vertices" in report:
        lines.append(f"Cloud vertices   {report['cloud_vertices']:,}")
    for key, unit, name in (
        ("disparity_px", "px", "Disparity"),
        ("depth_m", "m", "Depth"),
    ):
        metrics = report.get("metrics", {}).get(key)
        if metrics and metrics["mae"] is not None:
            lines.append(
                f"{name} MAE   {metrics['mae']:.3f} {unit} "
                f"({metrics['coverage_pct']:.1f}% coverage)"
            )
    return "\n".join(lines)


def show_reconstruction_summary(
    output: Path,
    inputs: tuple[Image, Image],
    alignment: Image,
    disparity: FloatMap,
    depth: FloatMap | None,
    report: dict[str, Any],
) -> None:
    """Save and display a visual summary of a completed stereo run.

    Args:
        output: Run directory containing the saved maps and optional PLY.
        inputs: Original left and right BGR images before rectification.
        alignment: Rectified views with horizontal epipolar guides.
        disparity: Estimated disparity in pixels.
        depth: Metric depth in metres, when calibration supports it.
        report: The metadata saved in run.json.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 9), layout="constrained")
    scene_path = report.get("inputs", {}).get("scene")
    title = Path(scene_path).name if scene_path else "Raw stereo pair"
    figure.suptitle(f"Stereo reconstruction · {title}")
    figure.canvas.manager.set_window_title(title)
    grid = figure.add_gridspec(3, 3, width_ratios=(1, 1, 0.9))

    for slot, title, image in (
        (grid[0, 0], "Left input", inputs[0]),
        (grid[0, 1], "Right input", inputs[1]),
        (grid[2, :2], "Rectified views · horizontal guides", alignment),
    ):
        axis = figure.add_subplot(slot)
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.set_axis_off()

    # Keep the color scale consistent with the separately exported previews.
    palette = plt.get_cmap("turbo").with_extremes(bad="black")
    for column, title, key, unit, values in (
        (0, "Disparity", "disparity_px", "px", disparity),
        (1, "Depth", "depth_m", "m", depth),
    ):
        axis = figure.add_subplot(grid[1, column])
        axis.set_title(title)
        axis.set_axis_off()
        bounds = report["preview_ranges"].get(key)
        if values is None or bounds is None:
            message = (
                "Metric calibration required" if values is None else "No valid pixels"
            )
            axis.text(
                0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes
            )
            continue
        lower, upper = bounds
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        plotted = axis.imshow(
            np.ma.masked_invalid(values), cmap=palette, vmin=lower, vmax=upper
        )
        figure.colorbar(plotted, ax=axis, label=unit, fraction=0.046, pad=0.03)

    axis = figure.add_subplot(grid[:2, 2])
    axis.set_title("Run metadata", loc="left")
    axis.set_axis_off()
    axis.text(
        0.0,
        0.94,
        _summary_text(report),
        va="top",
        linespacing=1.6,
        transform=axis.transAxes,
    )

    cloud_path = output / "cloud_m.ply"
    if cloud_path.is_file():
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(cloud_path))
        xyz, rgb = np.asarray(cloud.points), np.asarray(cloud.colors)
        # A few thousand points are enough to show the scene shape clearly.
        stride = max(1, (len(xyz) + 2499) // 2500)
        axis = figure.add_subplot(grid[2, 2], projection="3d")
        axis.scatter(*xyz[::stride].T, c=rgb[::stride], s=2, depthshade=False)
        axis.set(
            xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Point cloud preview"
        )
        axis.invert_yaxis()
        axis.set_box_aspect((1, 1, 1), zoom=0.8)
        axis.tick_params(labelsize=7)

    _save_and_show(figure, output / "summary.png")


def show_calibration_summary(
    path: Path, left: Image, right: Image, metadata: str
) -> None:
    """Display a detected chessboard pair and save its calibration summary."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        layout="constrained",
        gridspec_kw={"width_ratios": (1, 1, 0.9)},
    )
    figure.suptitle("Stereo calibration")
    for axis, image, title in zip(
        axes[:2], (left, right), ("Left corners", "Right corners")
    ):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.set_axis_off()
    axes[2].set_title("Fit metadata", loc="left")
    axes[2].set_axis_off()
    axes[2].text(
        0.0, 0.9, metadata, va="top", transform=axes[2].transAxes, linespacing=1.6
    )
    _save_and_show(figure, path)
