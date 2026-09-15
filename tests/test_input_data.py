"""Input dataset structure and image-integrity tests."""

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import pytest
from torchvision.datasets import ImageFolder
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AREAS = ("PLeft", "PRight", "RoboArm", "ConvBelt")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def write_rgb(path: Path, value: int = 127, size=(128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


def inspect_image(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image.convert("RGB"))
        return {
            "source_mode": image.mode,
            "size": image.size,
            "all_black": bool(np.max(array) == 0),
            "finite": bool(np.isfinite(array).all()),
        }


def test_imagefolder_accepts_expected_area_class_layout(tmp_path):
    area_root = tmp_path / "PLeft"
    write_rgb(area_root / "normal" / "frame_000001.png")
    write_rgb(area_root / "normal" / "frame_000002.png", value=200)

    dataset = ImageFolder(area_root)

    assert dataset.classes == ["normal"]
    assert dataset.class_to_idx == {"normal": 0}
    assert len(dataset) == 2


def test_rgb_image_integrity_and_dimensions(tmp_path):
    path = tmp_path / "normal" / "frame.png"
    write_rgb(path, size=(128, 128))

    result = inspect_image(path)

    assert result["source_mode"] == "RGB"
    assert result["size"] == (128, 128)
    assert result["finite"] is True
    assert result["all_black"] is False


def test_black_frame_is_detected(tmp_path):
    path = tmp_path / "normal" / "black.png"
    write_rgb(path, value=0)

    assert inspect_image(path)["all_black"] is True


def test_corrupt_image_is_rejected(tmp_path):
    path = tmp_path / "normal" / "broken.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not an image")

    with pytest.raises((UnidentifiedImageError, OSError)):
        inspect_image(path)


def test_train_and_test_paths_do_not_overlap(tmp_path):
    train = tmp_path / "train" / "PLeft" / "normal" / "frame_1.png"
    test = tmp_path / "test" / "PLeft" / "normal" / "frame_2.png"
    write_rgb(train)
    write_rgb(test)

    train_names = {path.name for path in train.parents[2].rglob("*.png")}
    test_names = {path.name for path in test.parents[2].rglob("*.png")}

    assert train_names.isdisjoint(test_names)


def test_configured_training_data_samples_are_readable_when_available():
    """Small Epito integration check; skipped on machines without that dataset."""
    config_path = REPOSITORY_ROOT / "configs/cf_dataset_epito.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training_root = Path(config["data"]["training"]).expanduser()
    if not training_root.is_dir():
        pytest.skip(f"configured training dataset unavailable: {training_root}")

    for area in AREAS:
        area_root = training_root / area
        assert area_root.is_dir(), f"missing safety-area directory: {area_root}"
        samples = sorted(
            path for path in area_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )[:5]
        assert samples, f"no images found for {area}"
        for sample in samples:
            result = inspect_image(sample)
            assert result["source_mode"] == "RGB", f"non-RGB input: {sample}"
            assert result["finite"]
            assert not result["all_black"], f"completely black input: {sample}"
