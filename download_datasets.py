"""Download the OpenCV stereo calibration samples and Middlebury Eval3 data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile, is_zipfile

import requests

LOGGER = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"
MIDDLEBURY_URL = "https://vision.middlebury.edu/stereo/submit3/zip/"
OPENCV_URL = "https://raw.githubusercontent.com/opencv/opencv/4.11.0/samples/data/"


def download_file(url: str, destination: Path, *, archive: bool = False) -> None:
    """Stream a download to a temporary file and publish it after validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        if not archive or is_zipfile(destination):
            return

    LOGGER.info("Downloading %s", destination.name)
    with NamedTemporaryFile(
        dir=destination.parent, suffix=".part", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
        if temporary.stat().st_size == 0 or (archive and not is_zipfile(temporary)):
            raise ValueError(f"Unexpected download format: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_middlebury(*, ground_truth: bool = False) -> None:
    """Extract either the quarter-size inputs or the training reference archive."""
    directory = DATA_DIR / "middlebury"
    name = "MiddEval3-GT0-Q.zip" if ground_truth else "MiddEval3-data-Q.zip"
    files = (
        ("disp0GT.pfm", "mask0nocc.png")
        if ground_truth
        else (
            "im0.png",
            "im1.png",
            "calib.txt",
        )
    )
    splits = ("trainingQ",) if ground_truth else ("trainingQ", "testQ")
    complete = True
    for split in splits:
        scenes = [
            path
            for path in (directory / "MiddEval3" / split).glob("*")
            if path.is_dir()
        ]
        complete &= (
            sum(
                all(
                    (scene / file).is_file() and (scene / file).stat().st_size > 0
                    for file in files
                )
                for scene in scenes
            )
            == 15
        )
    if complete:
        LOGGER.info("Already extracted %s", name)
        return

    path = directory / name
    download_file(MIDDLEBURY_URL + name, path, archive=True)
    with ZipFile(path) as download:
        download.extractall(directory)
    LOGGER.info("Extracted %s", name)


def download_chessboard() -> None:
    """Fetch paired chessboard images from the OpenCV stereo sample list."""
    for index in range(1, 15):
        if index == 10:
            continue
        for side in ("left", "right"):
            name = f"{side}{index:02d}.jpg"
            download_file(OPENCV_URL + name, DATA_DIR / "calibration" / side / name)
    LOGGER.info("OpenCV chessboard data ready")


def main(argv: list[str] | None = None) -> int:
    """Download selected datasets and return a conventional process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("middlebury", "chessboard", "ground-truth"),
        help="Download one component (default: images for both datasets)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.only in (None, "middlebury"):
            download_middlebury()
        if args.only in (None, "chessboard"):
            download_chessboard()
        if args.only == "ground-truth":
            download_middlebury(ground_truth=True)
    except (
        OSError,
        requests.RequestException,
        BadZipFile,
        ValueError,
        RuntimeError,
    ) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
