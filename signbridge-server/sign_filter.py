"""
sign_filter.py — 3-layer pre-filter for 30 ASL signs
Updated for correct dataset signs.

Sign references (Lifeprint.com / ASL University):
---------------------------------------------------
hello    - flat open hand, salute from forehead outward, sideways wave
bye      - flat open hand, wave at ear/mouth level
yes      - fist nodding up and down at chest/neutral level
no       - index+middle extended, shake side to side
please   - flat hand on chest, circular motion
thankyou - flat hand from chin, moves outward and down
happy    - flat hand brushing up on chest repeatedly
sad      - open hands drop down from face level
sick     - middle finger touches forehead, other touches chest
hungry   - C-hand slides down chest
sleepy   - open 5-hand closes in front of face, drops
sleep    - open hand closes over face and drops down
drink    - C-shape hand tips toward mouth
go       - both index fingers point and arc forward
look     - V-hand points from eyes outward
think    - index finger circles at temple
finish   - open hands flip outward from chest
taste    - middle finger touches lips/chin
mom      - open 5-hand, thumb touches chin, tap twice
dad      - open 5-hand, thumb touches forehead, tap twice
girl     - thumb brushes down cheek (A-hand)
boy      - hand opens and closes at forehead (cap brim)
man      - open 5-hand touches forehead then chest
time     - index finger taps wrist (like tapping a watch)
home     - pinched O-hand touches cheek then chin
water    - W-shape (3 fingers), taps chin twice
food     - pinched fingers tap mouth
dog      - pat thigh then snap fingers
cat      - F-hand pinch pulls whiskers from cheek outward
bird     - G-hand at mouth, fingers open/close like beak
"""

from collections import deque
import numpy as np

# ── Sign definitions ──────────────────────────────────────────────────
# (location_zones, movement_types, required_fingers, excluded_fingers)

SIGN_DEFINITIONS = {
    # Greetings / responses
    "hello":    (["forehead",],               ["wave", "side", "outward"],  ["open_flat"],           ["fist", "curved"]),
    "bye":      (["ear_level"],              ["wave", "side"],             ["open_flat"],           ["fist", "curved"]),
    "yes":      (["chest", "neutral"],       ["nod", "tap", "down"],       ["fist"],                ["open_flat", "two_fingers", "three_fingers"]),
    "no":       (["neutral", "chest"],       ["side", "wave", "still"],    ["two_fingers"],         ["fist", "open_flat", "three_fingers"]),
    "please":   (["chest"],                  ["circular"],                 ["open_flat"],           ["fist", "two_fingers"]),
    "thankyou": (["chin", "mouth"],          ["outward", "down", "forward"],["open_flat"],          ["fist"]),

    # Feelings / states
    "happy":    (["chest"],                  ["up", "circular"],           ["open_flat"],           ["fist"]),
    "sad":      (["upper_face", "face"],     ["down"],                     ["open_flat"],           ["fist", "pinch"]),
    "sick":     (["upper_face", "face",
                  "chin", "chest"],          ["still", "touch"],           ["index", "curved"],     ["open_flat", "fist"]),
    "hungry":   (["chest"],                  ["down"],                     ["curved_C", "curved"],  ["open_flat", "fist"]),
    "sleepy":   (["upper_face", "face",
                  "neutral"],                ["down"],                     ["open_flat", "curved"],  ["fist"]),
    "sleep":    (["upper_face", "face"],     ["down"],                     ["open_flat", "curved"],  ["fist"]),

    # Actions
    "drink":    (["mouth", "chin"],          ["up", "still"],              ["curved_C", "curved"],  ["open_flat", "fist"]),
    "go":       (["neutral"],                ["outward", "forward"],       ["index", "two_fingers"], ["fist", "open_flat"]),
    "look":     (["upper_face", "face"],     ["outward", "forward"],       ["two_fingers"],         ["fist", "open_flat"]),
    "think":    (["upper_face", "face"],     ["circular", "still"],        ["index"],               ["fist", "open_flat"]),
    "finish":   (["neutral", "chest"],       ["outward", "side"],          ["open_flat"],           ["fist", "pinch"]),
    "taste":    (["mouth", "chin"],          ["tap", "still"],             ["index", "curved"],     ["open_flat", "fist"]),

    # People
    "mom":      (["chin", "mouth"],          ["tap", "still"],             ["open_flat"],           ["fist", "pinch"]),
    "dad":      (["forehead", "upper_face"], ["tap", "still"],             ["open_flat"],           ["fist", "pinch"]),
    "girl":     (["cheek", "chin"],          ["down", "outward"],          ["fist", "curved"],      ["open_flat", "two_fingers"]),
    "boy":      (["forehead", "upper_face"], ["still", "squeeze", "down"], ["curved", "open_flat"], ["fist"]),
    "man":      (["forehead", "upper_face",
                  "chest"],                  ["down", "touch"],            ["open_flat"],           ["fist", "pinch"]),
    "time":     (["neutral", "chest"],       ["tap", "still"],             ["index", "curved"],     ["open_flat", "fist"]),

    # Things / places
    "home":     (["cheek", "chin"],          ["tap", "still"],             ["pinch"],               ["open_flat", "fist", "two_fingers"]),
    "water":    (["chin", "mouth"],          ["tap", "nod", "down"],       ["three_fingers"],       ["fist", "open_flat"]),
    "food":     (["mouth"],                  ["tap", "still"],             ["pinch"],               ["open_flat", "fist"]),
    "dog":      (["neutral", "chest"],       ["tap", "down"],              ["open_flat", "index"],  ["fist"]),
    "cat":      (["cheek", "ear_level"],     ["outward", "side"],          ["pinch"],               ["open_flat", "fist"]),
    "bird":     (["mouth", "chin"],          ["still", "tap"],             ["pinch", "index"],      ["open_flat", "fist"]),
}

ALL_SIGNS = list(SIGN_DEFINITIONS.keys())


# ── Layer 1: Position ─────────────────────────────────────────────────
def classify_position(anchor):
    """
    Normalized Y values after shoulder normalization:
      forehead:   y < -0.90      (above eyes — hello, dad)
      upper_face: -1.00 to -0.70 (eye/temple — look, think, sad, sleep)
      face:       -0.90 to -0.65 (general face level)
      ear_level:  -0.80 to -0.45 (ear/mouth height — bye, cat)
      cheek/chin: -0.60 to -0.22 (mouth area — food, water, home, mom)
      chest:      -0.28 to  0.35 (chest — please, happy, hungry)
      neutral:    -0.50 to  0.35 (broad middle — yes, no, go, finish)
    """
    if not anchor or not anchor.get("active_wrist"):
        return set(["neutral"])

    w  = anchor["active_wrist"]
    wx = w["x"]
    wy = w["y"]
    zones = set()

    if wy < -0.90:
        zones.add("forehead")

    if -1.00 < wy < -0.70:
        zones.add("upper_face")
        zones.add("face")

    if -0.80 < wy < -0.45:
        zones.add("ear_level")

    if -0.60 < wy < -0.22:
        zones.add("cheek")
        zones.add("chin")
        zones.add("mouth")

    if -0.28 < wy < 0.35:
        zones.add("chest")

    if -0.50 < wy < 0.35:
        zones.add("neutral")

    if abs(wx) > 0.65:
        zones.add("side")

    return zones if zones else set(["neutral"])


# ── Layer 2: Movement ─────────────────────────────────────────────────
class MovementTracker:
    def __init__(self, window=20):
        self.history = deque(maxlen=window)

    def update(self, anchor):
        if anchor and anchor.get("active_wrist"):
            w = anchor["active_wrist"]
            self.history.append((w["x"], w["y"]))

    def classify(self):
        if len(self.history) < 6:
            return set(["still"])

        positions  = list(self.history)
        xs         = [p[0] for p in positions]
        ys         = [p[1] for p in positions]
        dx         = positions[-1][0] - positions[0][0]
        dy         = positions[-1][1] - positions[0][1]
        total_disp = np.sqrt(dx**2 + dy**2)
        avg_speed  = np.mean([
            np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
            for i in range(1, len(positions))
        ])

        movements = set()

        # Still — barely moving
        if total_disp < 0.04 and avg_speed < 0.007:
            movements.add("still")
            return movements

        # Directional
        if dy < -0.06: movements.add("up")
        if dy >  0.06: movements.add("down")
        if dx >  0.06:
            movements.add("outward")
            movements.add("forward")
        if dx < -0.06:
            movements.add("back")
            movements.add("outward")

        # Y reversals — tapping / nodding
        y_rev, last = 0, 0
        for i in range(1, len(ys)):
            d = ys[i] - ys[i-1]
            c = 1 if d > 0.004 else (-1 if d < -0.004 else 0)
            if c != 0 and c != last and last != 0:
                y_rev += 1
            if c != 0:
                last = c
        if y_rev >= 2:
            movements.add("tap")
            movements.add("nod")

        # X reversals — waving / shaking
        x_rev, last = 0, 0
        for i in range(1, len(xs)):
            d = xs[i] - xs[i-1]
            c = 1 if d > 0.004 else (-1 if d < -0.004 else 0)
            if c != 0 and c != last and last != 0:
                x_rev += 1
            if c != 0:
                last = c
        if x_rev >= 1:
            movements.add("wave")
            movements.add("side")

        # Circular — both x and y reversals
        if x_rev >= 1 and y_rev >= 1:
            movements.add("circular")

        # Touch / still-ish
        if total_disp > 0.02:
            movements.add("touch")
            movements.add("squeeze")

        return movements if movements else set(["still"])


# ── Layer 3: Finger state ─────────────────────────────────────────────
def classify_fingers(hand_landmarks):
    if not hand_landmarks or len(hand_landmarks) < 21:
        return set(["open_flat"])

    lm = hand_landmarks

    def angle_3d(a, b, c):
        v1 = np.array([a["x"]-b["x"], a["y"]-b["y"],
                        a.get("z",0)-b.get("z",0)])
        v2 = np.array([c["x"]-b["x"], c["y"]-b["y"],
                        c.get("z",0)-b.get("z",0)])
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 == 0 or m2 == 0:
            return 180.0
        return np.degrees(np.arccos(
            np.clip(np.dot(v1, v2) / (m1 * m2), -1, 1)))

    def straight(mcp, pip, dip, tip, t=140):
        return (angle_3d(lm[mcp], lm[pip], lm[dip]) +
                angle_3d(lm[pip], lm[dip], lm[tip])) / 2 > t

    idx_up = straight(5,  6,  7,  8)
    mid_up = straight(9,  10, 11, 12)
    rng_up = straight(13, 14, 15, 16)
    pnk_up = straight(17, 18, 19, 20)

    wrist     = lm[0]
    hand_size = max(0.01, np.sqrt(
        (wrist["x"]-lm[9]["x"])**2 +
        (wrist["y"]-lm[9]["y"])**2))
    thumb_tip = lm[4]
    thm_up    = np.sqrt(
        (thumb_tip["x"]-lm[5]["x"])**2 +
        (thumb_tip["y"]-lm[5]["y"])**2) > hand_size * 0.38

    pinch_dist = np.sqrt(
        (thumb_tip["x"]-lm[8]["x"])**2 +
        (thumb_tip["y"]-lm[8]["y"])**2)
    is_pinch = pinch_dist < hand_size * 0.32

    extended = sum([thm_up, idx_up, mid_up, rng_up, pnk_up])
    patterns = set()

    if extended >= 4:
        patterns.add("open_flat")
    if extended <= 1 and not thm_up:
        patterns.add("fist")
    if is_pinch:
        patterns.add("pinch")
    if 1 <= extended <= 3 and not is_pinch:
        patterns.add("curved")
        patterns.add("curved_C")
    if idx_up and mid_up and not rng_up and not pnk_up:
        patterns.add("two_fingers")
    if idx_up and not mid_up and not rng_up and not pnk_up:
        patterns.add("index")
    if idx_up and mid_up and rng_up and not pnk_up:
        patterns.add("three_fingers")

    return patterns if patterns else set(["curved"])


# ── Main filter with exclusions ───────────────────────────────────────
def apply_filter(position_zones, movement_types, finger_patterns):
    candidates = []

    for sign, (locs, moves, required, excluded) in SIGN_DEFINITIONS.items():

        # Layer 1 — position must match
        if not any(z in position_zones for z in locs):
            continue

        # Layer 2 — movement must match
        if not any(m in movement_types for m in moves):
            continue

        # Layer 3a — required finger pattern must be present
        if not any(f in finger_patterns for f in required):
            continue

        # Layer 3b — excluded finger patterns must NOT be present
        if any(f in finger_patterns for f in excluded):
            continue

        candidates.append(sign)

    # Never return empty — fall back to full list
    return candidates if candidates else ALL_SIGNS