"""
normalizer.py
==============
Normalizes MediaPipe Holistic landmarks so predictions are
independent of:
  - Camera distance (how far you sit)
  - Camera height (where the camera is placed)
  - Body size differences between people

Strategy:
  - Use LEFT_SHOULDER (11) and RIGHT_SHOULDER (12) as anchor points
  - Subtract the midpoint of both shoulders from all landmarks
  - Divide by the distance between shoulders (scale normalization)

After normalization:
  - Shoulder midpoint is always at (0, 0)
  - Shoulder width is always 1.0
  - All other landmarks are relative to this fixed reference

Pose landmark indices (MediaPipe):
  0  = NOSE
  11 = LEFT_SHOULDER
  12 = RIGHT_SHOULDER
  13 = LEFT_ELBOW
  14 = RIGHT_ELBOW
  15 = LEFT_WRIST
  16 = RIGHT_WRIST
  17 = LEFT_PINKY
  18 = RIGHT_PINKY
  19 = LEFT_INDEX
  20 = RIGHT_INDEX
  21 = LEFT_THUMB
  22 = RIGHT_THUMB
"""

import numpy as np

# Pose landmark indices we care about
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
NOSE           = 0

def normalize_landmarks(pose, left_hand, right_hand):
    """
    Normalizes all landmarks relative to shoulder position and width.

    Args:
        pose:       list of dicts [{x, y, z, visibility}, ...]  33 points
        left_hand:  list of dicts [{x, y, z}, ...]             21 points or []
        right_hand: list of dicts [{x, y, z}, ...]             21 points or []

    Returns:
        pose_norm, lh_norm, rh_norm — same format, normalized coordinates
        anchor — dict with normalization info for the filter layers
    """

    # ── Need at least both shoulders to normalize ────────────────
    if not pose or len(pose) < 13:
        return pose, left_hand, right_hand, None

    ls = pose[LEFT_SHOULDER]   # left shoulder
    rs = pose[RIGHT_SHOULDER]  # right shoulder

    # Midpoint between shoulders — our origin
    origin_x = (ls["x"] + rs["x"]) / 2
    origin_y = (ls["y"] + rs["y"]) / 2
    origin_z = (ls.get("z", 0) + rs.get("z", 0)) / 2

    # Shoulder width — our scale unit
    shoulder_width = np.sqrt(
        (ls["x"] - rs["x"]) ** 2 +
        (ls["y"] - rs["y"]) ** 2
    )

    # Avoid division by zero
    if shoulder_width < 0.001:
        return pose, left_hand, right_hand, None

    def norm_point(p):
        """Normalize a single landmark point"""
        return {
            "x": (p["x"] - origin_x) / shoulder_width,
            "y": (p["y"] - origin_y) / shoulder_width,
            "z": (p.get("z", 0) - origin_z) / shoulder_width,
            "visibility": p.get("visibility", 1.0),
        }

    def norm_hand(hand):
        """Normalize a full hand landmark list"""
        if not hand:
            return hand
        return [norm_point(p) for p in hand]

    # Normalize all landmark groups
    pose_norm  = [norm_point(p) for p in pose]
    lh_norm    = norm_hand(left_hand)
    rh_norm    = norm_hand(right_hand)

    # ── Compute useful anchor info for the filter layers ────────
    # After normalization, these positions are in shoulder-relative coords

    # Nose position (face reference)
    nose_y = pose_norm[NOSE]["y"] if len(pose_norm) > NOSE else -1.0

    # Wrist positions (normalized)
    lw = pose_norm[LEFT_WRIST]  if len(pose_norm) > LEFT_WRIST  else None
    rw = pose_norm[RIGHT_WRIST] if len(pose_norm) > RIGHT_WRIST else None

    # Which hand is active — use the one with landmarks
    active_wrist = None
    if right_hand and rw:
        active_wrist = rw
    elif left_hand and lw:
        active_wrist = lw

    # Shoulder midpoint in normalized space is always (0, 0)
    # Nose is typically around y = -1.0 to -1.5 (above shoulders)
    # Face level:    wrist y < -0.6
    # Mouth level:   wrist y between -0.6 and -0.3
    # Chest level:   wrist y between -0.3 and +0.3
    # Neutral space: wrist y between -0.5 and +0.1 (in front)

    anchor = {
        "origin_x":      origin_x,
        "origin_y":      origin_y,
        "shoulder_width": shoulder_width,
        "nose_y":        nose_y,
        "active_wrist":  active_wrist,
        "left_wrist":    lw,
        "right_wrist":   rw,
    }

    return pose_norm, lh_norm, rh_norm, anchor


def extract_frame_normalized(pose, left_hand, right_hand):
    """
    Full pipeline: normalize then extract 258-value frame vector.
    Returns (frame_vector, anchor) tuple.
    """
    pose_n, lh_n, rh_n, anchor = normalize_landmarks(pose, left_hand, right_hand)

    # ── Build 258-value vector (same format as training) ────────
    # Pose: 33 × 4 = 132
    if pose_n:
        xyz = np.array([
            [p["x"], p["y"], p.get("z", 0)]
            for p in pose_n
        ]).flatten()
        pose_arr = np.zeros(132)
        pose_arr[:min(len(xyz), 99)] = xyz[:99]
    else:
        pose_arr = np.zeros(132)

    # Left hand: 21 × 3 = 63
    if lh_n:
        xyz = np.array([
            [p["x"], p["y"], p.get("z", 0)]
            for p in lh_n
        ]).flatten()
        lh_arr = np.zeros(63)
        lh_arr[:min(len(xyz), 63)] = xyz[:63]
    else:
        lh_arr = np.zeros(63)

    # Right hand: 21 × 3 = 63
    if rh_n:
        xyz = np.array([
            [p["x"], p["y"], p.get("z", 0)]
            for p in rh_n
        ]).flatten()
        rh_arr = np.zeros(63)
        rh_arr[:min(len(xyz), 63)] = xyz[:63]
    else:
        rh_arr = np.zeros(63)

    frame = np.concatenate([pose_arr, lh_arr, rh_arr])
    return np.nan_to_num(frame, nan=0.0), anchor