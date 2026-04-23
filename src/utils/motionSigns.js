/**
 * motionSigns.js
 *
 * Classifies 5 ASL motion signs using movement patterns
 * from motionBuffer.js.
 *
 * Signs (source: Lifeprint.com / ASL University):
 *
 *  HELP   — Closed fist resting on open palm, both hands lift upward together
 *  PLEASE — Open hand flat on chest, moves in a circular motion
 *  STOP   — One flat hand, other hand chops down onto it (sharp downward motion)
 *  MORE   — Both hands with fingers pinched (O-shape), tap fingertips together twice
 *  WATER  — W-handshape (index+middle+ring up), tap chin area downward twice
 *
 * Note: Since we track one hand at a time, we simplify two-hand signs:
 *  HELP  → one hand fist moving upward
 *  MORE  → one hand pinched (thumb+index touch) tapping motion
 */

const MOTION_THRESHOLD = 0.75;

// ── Individual motion classifiers ───────────────────────────────────

/**
 * HELP — fist shape moving upward
 *
 * Real ASL: fist on palm, lift up. We detect: fist + clear upward movement.
 * The fist must be consistent across frames (avgFistScore high).
 */
function classifyHelp(motion) {
  if (!motion) return 0;

  const { movingUp, avgFistScore, totalMovement, avgFingers } = motion;

  // Must be moving upward
  if (!movingUp) return 0;

  // Hand should be mostly a fist
  if (avgFistScore < 45) return 0;

  // Other fingers mostly folded (not open hand)
  if (avgFingers.index || avgFingers.middle) return 0;

  // Must have meaningful upward movement
  if (totalMovement < 0.08) return 0;

  // Confidence scales with fist tightness and movement amount
  const fistConf = Math.min(1, avgFistScore / 70);
  const moveConf = Math.min(1, totalMovement / 0.15);
  return (fistConf + moveConf) / 2;
}

/**
 * PLEASE — open flat hand, circular motion on chest area
 *
 * Real ASL: flat hand on chest in circles. We detect:
 * all fingers extended + circular motion + hand near mid-frame (chest area).
 */
function classifyPlease(motion) {
  if (!motion) return 0;

  const { isCircular, avgFingers, avgFistScore, avgPalmY, totalMovement } = motion;

  // Must be circular
  if (!isCircular) return 0;

  // Hand should be open — all 4 fingers extended
  if (!avgFingers.index || !avgFingers.middle) return 0;

  // Not a fist
  if (avgFistScore > 40) return 0;

  // Hand should be in the lower half of the frame (chest area)
  if (avgPalmY < 0.35) return 0;

  // Enough movement to be a real circle
  if (totalMovement < 0.05) return 0;

  return Math.min(1, totalMovement / 0.1 + 0.4);
}

/**
 * STOP — sharp downward chopping motion with open hand
 *
 * Real ASL: one hand chops down onto other palm. We detect:
 * flat open hand + fast sharp downward movement.
 */
function classifyStop(motion) {
  if (!motion) return 0;

  const { movingDown, avgSpeed, totalMovement, avgFingers, avgFistScore, deltaY } = motion;

  // Must be moving downward
  if (!movingDown) return 0;

  // Should be a relatively fast motion (chop is quick)
  if (avgSpeed < 0.003) return 0;

  // Hand should be fairly open
  if (avgFistScore > 50) return 0;

  // Index and middle should be extended (open/flat hand)
  if (!avgFingers.index || !avgFingers.middle) return 0;

  // Clear downward displacement
  if (deltaY < 0.08) return 0;

  const speedConf = Math.min(1, avgSpeed / 0.006);
  const moveConf  = Math.min(1, deltaY / 0.15);
  return (speedConf + moveConf) / 2;
}

/**
 * MORE — pinched hand (thumb touches index tip) tapping motion
 *
 * Real ASL: both hands pinched together tapping. We detect:
 * thumb and index close together (pinch) + oscillating/tapping motion.
 */
function classifyMore(motion) {
  if (!motion) return 0;

  const { isTapping, isOscillating, avgFistScore, avgFingers, frames } = motion;

  // Must have tapping or oscillating motion
  if (!isTapping && !isOscillating) return 0;

  // Hand should be partially closed (pinch shape)
  // Not fully open, not fully closed fist
  if (avgFistScore < 20 || avgFistScore > 75) return 0;

  // Check if thumb and index fingertips are close together in recent frames
  // (pinch detection)
  const recentFrames = frames.slice(-15);
  const pinchCount = recentFrames.filter(f => {
    if (!f.tips?.thumb || !f.tips?.index) return false;
    const dx = f.tips.thumb.x - f.tips.index.x;
    const dy = f.tips.thumb.y - f.tips.index.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    return dist < 0.08; // thumb and index tips close together
  }).length;

  const pinchRatio = pinchCount / recentFrames.length;
  if (pinchRatio < 0.4) return 0;

  return Math.min(1, pinchRatio * 0.7 + 0.3);
}

/**
 * WATER — W-handshape (index+middle+ring extended) + tapping downward motion
 *
 * Real ASL: W shape (3 fingers) tapping chin area. We detect:
 * index+middle+ring extended + thumb+pinky folded + tapping motion.
 */
function classifyWater(motion) {
  if (!motion) return 0;

  const { isTapping, isOscillating, avgFingers } = motion;

  // Must be tapping
  if (!isTapping && !isOscillating) return 0;

  // W-handshape: index + middle + ring must be extended
  if (!avgFingers.index || !avgFingers.middle || !avgFingers.ring) return 0;

  // Thumb should NOT be extended (distinguishes W from open hand)
  if (avgFingers.thumb && avgFingers.pinky) return 0;

  // Pinky should be folded
  if (avgFingers.pinky) return 0;

  // Good W shape — high confidence
  return 0.9;
}

// ── Main classifier ─────────────────────────────────────────────────

/**
 * Runs all 5 motion classifiers against the buffer analysis.
 *
 * @param   {object} motion - output from buffer.analyze()
 * @returns {object|null}   - { word, confidence } or null if no match
 */
export function classifyMotionSign(motion) {
  if (!motion) return null;

  const candidates = [
    { word: "HELP",   score: classifyHelp(motion)   },
    { word: "PLEASE", score: classifyPlease(motion) },
    { word: "STOP",   score: classifyStop(motion)   },
    { word: "MORE",   score: classifyMore(motion)   },
    { word: "WATER",  score: classifyWater(motion)  },
  ];

  const best = candidates.reduce(
    (max, c) => (c.score > max.score ? c : max),
    { word: null, score: 0 }
  );

  if (best.score >= MOTION_THRESHOLD) {
    return { word: best.word, confidence: best.score };
  }

  return null;
}

export { MOTION_THRESHOLD };