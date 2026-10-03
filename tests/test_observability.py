import numpy as np
from PIL import Image

from app.drift import check_drift
from app.monitoring import get_prediction_metrics
from app.system_monitoring import get_system_metrics


# ============================================================
# DATA DRIFT TEST
# ============================================================

def test_reference_like_image_has_no_drift():

    # Create a synthetic 64x64 RGB image whose
    # channel values are close to the EuroSAT
    # reference means.
    reference_rgb = np.array(
        [
            [0.34437728, 0.38029137, 0.40777302]
        ],
        dtype=np.float32,
    )

    pixel_values = (
        reference_rgb * 255
    ).astype(np.uint8)

    image_array = np.tile(
        pixel_values,
        (64, 64, 1),
    )

    image = Image.fromarray(
        image_array,
        mode="RGB",
    )

    result = check_drift(image)

    assert result["drift_detected"] is False
    assert result["drifted_features"] == []
    assert result["image_size_changed"] is False


# ============================================================
# MODEL MONITORING TEST
# ============================================================

def test_prediction_metrics_structure():

    metrics = get_prediction_metrics()

    required_keys = [
        "total_predictions",
        "labeled_predictions",
        "correct_predictions",
        "accuracy",
        "average_confidence",
        "average_latency_ms",
        "prediction_counts",
    ]

    for key in required_keys:
        assert key in metrics


# ============================================================
# INFRASTRUCTURE MONITORING TEST
# ============================================================

def test_system_metrics_structure():

    metrics = get_system_metrics()

    required_keys = [
        "cpu_percent",
        "memory_percent",
        "memory_used_mb",
        "memory_available_mb",
        "process_memory_mb",
        "disk_percent",
        "disk_free_gb",
        "uptime_seconds",
    ]

    for key in required_keys:
        assert key in metrics


def test_system_metrics_values_are_valid():

    metrics = get_system_metrics()

    assert 0 <= metrics["cpu_percent"] <= 100
    assert 0 <= metrics["memory_percent"] <= 100
    assert 0 <= metrics["disk_percent"] <= 100

    assert metrics["memory_used_mb"] >= 0
    assert metrics["memory_available_mb"] >= 0
    assert metrics["process_memory_mb"] >= 0
    assert metrics["disk_free_gb"] >= 0
    assert metrics["uptime_seconds"] >= 0