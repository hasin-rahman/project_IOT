"""Shared constants. The Kotlin side (Constants.kt) must mirror every value here."""
import math

# Class index order is part of the contract (model output k == POSES[k]).
POSES = [
    "screen_up",
    "screen_down",
    "portrait_up",
    "portrait_down",
    "landscape_left",
    "landscape_right",
]
POSE_INDEX = {p: i for i, p in enumerate(POSES)}
N_CLASSES = len(POSES)

# Non-IID split from the proposal: A trains on the first three poses, B on the last three.
CLIENT_POSES = {"A": (0, 1, 2), "B": (3, 4, 5)}
CLIENT_INDEX = {"A": 0, "B": 1}

SENSORS = ("acc", "grav", "gyro", "rot")

# Fixed (client independent) scaling. We do NOT standardize per client, because
# per-client mean/std would differ under non-IID data and break FedAvg.
SENSOR_SCALE = {"acc": 9.81, "grav": 9.81, "gyro": math.pi, "rot": 1.0}

# Sensor study (required technical extension). The baseline is "acc_grav" (12 features).
FEATURE_SETS = {
    "acc": ("acc",),
    "acc_grav": ("acc", "grav"),
    "acc_grav_gyro": ("acc", "grav", "gyro"),
    "acc_grav_gyro_rot": ("acc", "grav", "gyro", "rot"),
}
DEFAULT_FEATURE_SET = "acc_grav"

# Windowing: 1 s windows, 50 % overlap, ~50 Hz sensors.
WINDOW_S = 1.0
STRIDE_S = 0.5
TRIM_S = 1.0  # seconds dropped at both ends of a session (hand placement motion)
MIN_SAMPLES = 10  # minimum samples per sensor inside a window

SPLITS = ("train", "val", "test", "newuser")


def n_features(feature_set: str) -> int:
    """Each sensor contributes mean(x,y,z) + std(x,y,z) = 6 features."""
    return 6 * len(FEATURE_SETS[feature_set])


def n_params(nf: int) -> int:
    """Softmax regression: 6 x nf weights + 6 biases (78 for nf = 12)."""
    return N_CLASSES * nf + N_CLASSES
