/**
 * signGeometry.js
 *
 * Uses JOINT ANGLES instead of Y-axis comparison.
 * This works correctly regardless of hand orientation —
 * facing camera, pointing up, sideways, all work the same.
 *
 * MediaPipe landmark indices:
 *   0  = WRIST
 *   1-4  = THUMB  (CMC, MCP, IP, TIP)
 *   5-8  = INDEX  (MCP, PIP, DIP, TIP)
 *   9-12 = MIDDLE (MCP, PIP, DIP, TIP)
 *   13-16= RING   (MCP, PIP, DIP, TIP)
 *   17-20= PINKY  (MCP, PIP, DIP, TIP)
 */

export const LM = {
  WRIST:      0,
  THUMB_CMC:  1, THUMB_MCP:  2, THUMB_IP:   3, THUMB_TIP:  4,
  INDEX_MCP:  5, INDEX_PIP:  6, INDEX_DIP:  7, INDEX_TIP:  8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10,MIDDLE_DIP: 11,MIDDLE_TIP: 12,
  RING_MCP:   13,RING_PIP:   14,RING_DIP:   15,RING_TIP:   16,
  PINKY_MCP:  17,PINKY_PIP:  18,PINKY_DIP:  19,PINKY_TIP:  20,
};

// ── Vector math ────────────────────────────────────────────────────

export function distance(a, b) {
  return Math.sqrt(
    Math.pow(a.x - b.x, 2) +
    Math.pow(a.y - b.y, 2) +
    Math.pow((a.z || 0) - (b.z || 0), 2)
  );
}

/**
 * Angle at joint B, formed by points A-B-C.
 * Uses all 3 axes (x, y, z) so orientation doesn't matter.
 * Returns degrees 0-180.
 */
function angleBetween3D(a, b, c) {
  const v1 = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: (a.z || 0) - (b.z || 0),
  };
  const v2 = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: (c.z || 0) - (b.z || 0),
  };

  const dot  = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
  const mag1 = Math.sqrt(v1.x**2 + v1.y**2 + v1.z**2);
  const mag2 = Math.sqrt(v2.x**2 + v2.y**2 + v2.z**2);

  if (mag1 === 0 || mag2 === 0) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2)))) * (180 / Math.PI);
}

// ── Finger curl measurement ─────────────────────────────────────────

/**
 * Measures how straight a finger is by averaging:
 * - Angle at MCP joint (base knuckle)
 * - Angle at PIP joint (middle knuckle)
 *
 * Straight finger → both angles close to 180° → high straightness
 * Curled finger   → both angles small          → low straightness
 *
 * Returns 0 (fully curled) to 180 (fully straight)
 */
function getFingerStraightness(lm, mcp, pip, dip, tip) {
  const mcpAngle = angleBetween3D(lm[LM.WRIST], lm[mcp], lm[pip]);
  const pipAngle = angleBetween3D(lm[mcp],      lm[pip], lm[dip]);
  const dipAngle = angleBetween3D(lm[pip],      lm[dip], lm[tip]);
  return (mcpAngle + pipAngle + dipAngle) / 3;
}

/**
 * Is this finger extended?
 * Threshold: average joint angle > 145° = extended
 * This threshold works well across hand orientations.
 */
function isFingerExtended(lm, mcp, pip, dip, tip, threshold = 145) {
  const straightness = getFingerStraightness(lm, mcp, pip, dip, tip);
  return straightness > threshold;
}

/**
 * Thumb is special — it moves on a different axis.
 * We check the angle at the IP joint AND the distance
 * of thumb tip from the index finger base.
 */
function isThumbExtended(lm) {
  // Angle at thumb IP joint
  const ipAngle = angleBetween3D(
    lm[LM.THUMB_MCP],
    lm[LM.THUMB_IP],
    lm[LM.THUMB_TIP]
  );

  // Angle at thumb MCP joint
  const mcpAngle = angleBetween3D(
    lm[LM.THUMB_CMC],
    lm[LM.THUMB_MCP],
    lm[LM.THUMB_IP]
  );

  // Distance from thumb tip to index MCP — thumb extended = far away
  const thumbToIndex = distance(lm[LM.THUMB_TIP], lm[LM.INDEX_MCP]);
  const handSize     = distance(lm[LM.WRIST], lm[LM.MIDDLE_MCP]);

  const anglesOk = (ipAngle + mcpAngle) / 2 > 140;
  const distOk   = thumbToIndex > handSize * 0.4;

  // Both conditions must pass to avoid false positives
  return anglesOk && distOk;
}

// ── Fist score ─────────────────────────────────────────────────────

/**
 * Returns 0-100 representing how closed the fist is.
 *
 * Uses XY-only distance from each fingertip to its own MCP base joint.
 * MediaPipe X and Y are accurate — Z is not, so we ignore Z here.
 *
 * Open hand  → tip far from MCP   → low score
 * Tight fist → tip close to MCP   → high score
 *
 * We normalize by hand size so it works at any distance from camera.
 */
function getFistScore(lm) {
  // Use PIP joint angles — the middle knuckle bends the most in a fist.
  // This works in any hand orientation including facing the camera.
  //
  // PIP angle when open: ~160-170°
  // PIP angle when fist: ~40-70°
  //
  // We use all 3 axes so camera angle doesn't affect the reading.

  const pipAngles = [
    angleBetween3D(lm[LM.INDEX_MCP],  lm[LM.INDEX_PIP],  lm[LM.INDEX_DIP]),
    angleBetween3D(lm[LM.MIDDLE_MCP], lm[LM.MIDDLE_PIP], lm[LM.MIDDLE_DIP]),
    angleBetween3D(lm[LM.RING_MCP],   lm[LM.RING_PIP],   lm[LM.RING_DIP]),
    angleBetween3D(lm[LM.PINKY_MCP],  lm[LM.PINKY_PIP],  lm[LM.PINKY_DIP]),
  ];

  const avgPip = pipAngles.reduce((a, b) => a + b, 0) / pipAngles.length;

  // Map: 165° (open) → 0%,   50° (fist) → 100%
  const OPEN = 165;
  const FIST = 50;
  const score = ((OPEN - avgPip) / (OPEN - FIST)) * 100;
  return Math.round(Math.max(0, Math.min(100, score)));
}

/**
 * Raw average straightness — exported for debug panel calibration.
 * Not used in final sign detection.
 */
export function getRawStraightness(lm) {
  const fingers = [
    getFingerStraightness(lm, LM.INDEX_MCP,  LM.INDEX_PIP,  LM.INDEX_DIP,  LM.INDEX_TIP),
    getFingerStraightness(lm, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    getFingerStraightness(lm, LM.RING_MCP,   LM.RING_PIP,   LM.RING_DIP,   LM.RING_TIP),
    getFingerStraightness(lm, LM.PINKY_MCP,  LM.PINKY_PIP,  LM.PINKY_DIP,  LM.PINKY_TIP),
  ];
  return Math.round(fingers.reduce((a, b) => a + b, 0) / fingers.length);
}

// ── Helpers ────────────────────────────────────────────────────────

export function getHandSize(lm) {
  return distance(lm[LM.WRIST], lm[LM.MIDDLE_MCP]);
}

export function getPalmCenter(lm) {
  const mcps = [LM.INDEX_MCP, LM.MIDDLE_MCP, LM.RING_MCP, LM.PINKY_MCP];
  const sum  = mcps.reduce(
    (acc, i) => ({ x: acc.x + lm[i].x, y: acc.y + lm[i].y }),
    { x: 0, y: 0 }
  );
  return { x: sum.x / 4, y: sum.y / 4 };
}

export function areFingersClose(lm, tip1, tip2) {
  return distance(lm[tip1], lm[tip2]) < getHandSize(lm) * 0.25;
}

// ── Main export ────────────────────────────────────────────────────

/**
 * Full geometry snapshot from one hand's landmarks.
 * Call once per frame, pass result to sign classifiers.
 */
export function getHandGeometry(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  const lm = landmarks;

  const fingers = {
    thumb:  isThumbExtended(lm),
    index:  isFingerExtended(lm, LM.INDEX_MCP,  LM.INDEX_PIP,  LM.INDEX_DIP,  LM.INDEX_TIP),
    middle: isFingerExtended(lm, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    ring:   isFingerExtended(lm, LM.RING_MCP,   LM.RING_PIP,   LM.RING_DIP,   LM.RING_TIP),
    pinky:  isFingerExtended(lm, LM.PINKY_MCP,  LM.PINKY_PIP,  LM.PINKY_DIP,  LM.PINKY_TIP),
  };

  return {
    lm,
    fingers,
    fistScore:   getFistScore(lm),
    handSize:    getHandSize(lm),
    palmCenter:  getPalmCenter(lm),
    tips: {
      thumb:  lm[LM.THUMB_TIP],
      index:  lm[LM.INDEX_TIP],
      middle: lm[LM.MIDDLE_TIP],
      ring:   lm[LM.RING_TIP],
      pinky:  lm[LM.PINKY_TIP],
    },
    wrist: lm[LM.WRIST],
  };
}