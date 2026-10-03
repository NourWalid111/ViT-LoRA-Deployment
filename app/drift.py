import numpy as np
from PIL import Image


# ============================================================
# REFERENCE DATASET PROFILE
# ============================================================
#
# These values were calculated from all 27,000 EuroSAT images.
#
# For each feature we store:
#   mean = average across the dataset
#   std  = variation across the dataset
#
# This lets us determine whether an incoming image is
# unusually different from the reference population.
# ============================================================

REFERENCE_PROFILE = {
    "red_mean": {
        "mean": 0.34437728,
        "std": 0.09136751,
    },

    "green_mean": {
        "mean": 0.38029137,
        "std": 0.06511851,
    },

    "blue_mean": {
        "mean": 0.40777302,
        "std": 0.05524026,
    },

    "red_std": {
        "mean": 0.09136751,
        "std": 0.0,
    },

    "green_std": {
        "mean": 0.06511851,
        "std": 0.0,
    },

    "blue_std": {
        "mean": 0.05524026,
        "std": 0.0,
    },

    "brightness_mean": {
        "mean": 0.37747902,
        "std": 0.12074776,
    },

    "brightness_std": {
        "mean": 0.0,
        "std": 0.0,
    },

    "image_height": 64,
    "image_width": 64,
}


# ============================================================
# DRIFT THRESHOLD
# ============================================================

Z_SCORE_THRESHOLD = 3.0


# ============================================================
# IMAGE STATISTICS
# ============================================================

def calculate_image_statistics(image: Image.Image):
    """
    Calculate basic statistics for one RGB image.
    """

    image_array = np.array(
        image.convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

    return {
        "red_mean": float(red.mean()),
        "green_mean": float(green.mean()),
        "blue_mean": float(blue.mean()),

        "red_std": float(red.std()),
        "green_std": float(green.std()),
        "blue_std": float(blue.std()),

        "brightness_mean": float(
            image_array.mean()
        ),

        "brightness_std": float(
            image_array.std()
        ),

        "image_height": image_array.shape[0],
        "image_width": image_array.shape[1],
    }


# ============================================================
# Z-SCORE
# ============================================================

def calculate_z_score(
    current_value,
    reference_mean,
    reference_std,
):
    """
    Calculate how many standard deviations an
    incoming value is from the reference mean.
    """

    if reference_std == 0:
        return 0.0

    return abs(
        current_value - reference_mean
    ) / reference_std


# ============================================================
# DRIFT CHECK
# ============================================================

def check_drift(image: Image.Image):
    """
    Compare an incoming image against the
    EuroSAT reference distribution.
    """

    current = calculate_image_statistics(
        image
    )

    z_scores = {}

    for feature in [
        "red_mean",
        "green_mean",
        "blue_mean",
        "brightness_mean",
    ]:

        reference = REFERENCE_PROFILE[
            feature
        ]

        z_scores[feature] = calculate_z_score(
            current[feature],
            reference["mean"],
            reference["std"],
        )

    size_changed = (
        current["image_height"]
        != REFERENCE_PROFILE["image_height"]
        or
        current["image_width"]
        != REFERENCE_PROFILE["image_width"]
    )

    drifted_features = [
        feature
        for feature, score in z_scores.items()
        if score > Z_SCORE_THRESHOLD
    ]

    drift_detected = (
        len(drifted_features) > 0
        or size_changed
    )

    return {
        "drift_detected": drift_detected,

        "drifted_features": drifted_features,

        "z_scores": z_scores,

        "current_statistics": current,

        "reference_statistics": REFERENCE_PROFILE,

        "image_size_changed": size_changed,
    }