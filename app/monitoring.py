import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONITORING_DIR = PROJECT_ROOT / "monitoring"
PREDICTIONS_FILE = MONITORING_DIR / "predictions.jsonl"


def _ensure_monitoring_directory():
    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def log_prediction(
    filename,
    predicted_class,
    confidence,
    latency_ms,
    true_label=None,
    drift_result=None,
):
    """
    Store one model prediction as a JSON Lines record.
    """

    _ensure_monitoring_directory()

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "filename": filename,

        "predicted_class": predicted_class,

        "confidence": round(
            float(confidence),
            4,
        ),

        "latency_ms": round(
            float(latency_ms),
            2,
        ),

        "true_label": true_label,

        "drift": drift_result,
    }

    with PREDICTIONS_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(record)
            + "\n"
        )

    return record


def load_predictions():
    """
    Load all stored prediction records.
    """

    if not PREDICTIONS_FILE.exists():
        return []

    records = []

    with PREDICTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def get_prediction_metrics():
    """
    Calculate basic model-performance metrics
    from predictions that have true labels.
    """

    records = load_predictions()

    labeled_records = [
        record
        for record in records
        if record.get("true_label") is not None
    ]

    correct = sum(
        record["predicted_class"]
        == record["true_label"]
        for record in labeled_records
    )

    accuracy = (
        correct / len(labeled_records)
        if labeled_records
        else None
    )

    average_confidence = (
        sum(
            record["confidence"]
            for record in records
        )
        / len(records)
        if records
        else 0.0
    )

    average_latency_ms = (
        sum(
            record["latency_ms"]
            for record in records
        )
        / len(records)
        if records
        else 0.0
    )

    prediction_counts = Counter(
        record["predicted_class"]
        for record in records
    )

    return {
        "total_predictions": len(records),

        "labeled_predictions": len(
            labeled_records
        ),

        "correct_predictions": correct,

        "accuracy": (
            round(accuracy, 4)
            if accuracy is not None
            else None
        ),

        "average_confidence": round(
            average_confidence,
            4,
        ),

        "average_latency_ms": round(
            average_latency_ms,
            2,
        ),

        "prediction_counts": dict(
            prediction_counts
        ),
    }


def measure_time():
    """
    Return a high-resolution timer value
    for measuring inference latency.
    """

    return time.perf_counter()