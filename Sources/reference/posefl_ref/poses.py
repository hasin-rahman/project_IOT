"""Pose definitions shared by every part of the project.

Android device axes: +x points to the right edge, +y to the top edge, +z out of the screen.
A resting accelerometer / gravity sensor reads +9.81 along whichever axis points UP.
"""

POSES = [
    # index, key, label, axis, sign
    (0, "screen_up", "Screen Up", 2, +1),
    (1, "screen_down", "Screen Down", 2, -1),
    (2, "portrait_up", "Portrait Up", 1, +1),
    (3, "portrait_down", "Portrait Down", 1, -1),
    (4, "landscape_left", "Landscape Left", 0, -1),
    (5, "landscape_right", "Landscape Right", 0, +1),
]

N_CLASSES = 6
KEYS = [p[1] for p in POSES]
LABELS = [p[2] for p in POSES]

CLIENT_TRAIN_POSES = {
    "A": [0, 1, 2],  # screen up, screen down, portrait up
    "B": [3, 4, 5],  # portrait down, landscape left, landscape right
}


def up_vector(pose_index):
    """Unit vector (device coordinates) that points up when the phone is in this pose."""
    _, _, _, axis, sign = POSES[pose_index]
    v = [0.0, 0.0, 0.0]
    v[axis] = float(sign)
    return v
