import pytest

from config import (
    DetectionConfig,
    FileConfig,
    VisualizationConfig,
    detection_config,
    validate_config,
)


def test_validate_config_passes():
    validate_config()


def test_confidence_threshold_in_range():
    assert 0.0 <= DetectionConfig.DEFAULT_CONFIDENCE_THRESHOLD <= 1.0
    assert 0.0 <= DetectionConfig.HIGH_CONFIDENCE_THRESHOLD <= 1.0
    assert 0.0 <= DetectionConfig.LOW_CONFIDENCE_THRESHOLD <= 1.0


def test_mask_alpha_in_range():
    assert 0.0 <= VisualizationConfig.MASK_ALPHA <= 1.0
    assert 0.0 <= VisualizationConfig.IMAGE_ALPHA <= 1.0


def test_vehicle_classes_have_korean_names():
    en_keys = set(DetectionConfig.VEHICLE_CLASSES.keys())
    kr_keys = set(DetectionConfig.CLASS_NAMES_KR.keys())
    assert en_keys == kr_keys


def test_vehicle_classes_have_colors():
    en_keys = set(DetectionConfig.VEHICLE_CLASSES.keys())
    color_keys = set(DetectionConfig.CLASS_COLORS.keys())
    assert en_keys == color_keys


def test_supported_image_formats_are_lowercase_with_dot():
    for ext in FileConfig.SUPPORTED_IMAGE_FORMATS:
        assert ext.startswith(".")
        assert ext == ext.lower()


def test_global_config_instances_exist():
    assert detection_config.DEFAULT_CONFIDENCE_THRESHOLD == DetectionConfig.DEFAULT_CONFIDENCE_THRESHOLD
