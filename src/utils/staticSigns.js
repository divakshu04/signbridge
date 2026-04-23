/**
 * staticSigns.js
 *
 * Classifies 5 static ASL hand signs using finger states
 * from signGeometry.js.
 *
 * Signs and their hand shapes (from Lifeprint.com / ASL University):
 *
 *  HELLO      — Open flat hand, all 5 fingers extended, palm out (like a salute/wave)
 *  YES        — Closed fist, all fingers curled tight, thumb alongside fist (nod gesture)
 *  NO         — Index + middle fingers extended together, rest folded (like a two-finger point)
 *  SORRY      — Thumb up, other 4 fingers folded into fist (A-handshape)
 *  I LOVE YOU — Thumb + index + pinky extended, middle + ring folded (ILY handshape)
 *
 * Each classifier returns a confidence 0.0 - 1.0.
 * A sign is only accepted if confidence >= CONFIDENCE_THRESHOLD.
 */

const CONFIDENCE_THRESHOLD = 0.80;

// ── Individual sign classifiers ─────────────────────────────────────

/**
 * HELLO — all 5 fingers extended, open hand
 * Finger pattern: thumb✓ index✓ middle✓ ring✓ pinky✓
 * Fist score must be low (hand is open)
 */
function classifyHello(geometry) {
  const { fingers, fistScore } = geometry;

  // All 5 must be extended
  const allExtended =
    fingers.thumb &&
    fingers.index &&
    fingers.middle &&
    fingers.ring &&
    fingers.pinky;

  if (!allExtended) return 0;

  // Hand should be open — penalize if fist score is high
  if (fistScore > 35) return 0;

  // Confidence based on how open the hand is
  const openness = Math.max(0, 1 - fistScore / 35);
  return openness;
}

/**
 * YES — tight fist, all fingers folded, thumb NOT extended
 * Finger pattern: thumb✗ index✗ middle✗ ring✗ pinky✗
 * Fist score must be high
 */
function classifyYes(geometry) {
  const { fingers, fistScore } = geometry;

  // All fingers must be folded
  const allFolded =
    !fingers.index &&
    !fingers.middle &&
    !fingers.ring &&
    !fingers.pinky;

  if (!allFolded) return 0;

  // Thumb should NOT be extended — distinguishes YES from SORRY
  if (fingers.thumb) return 0;

  // Fist must be reasonably tight
  if (fistScore < 50) return 0;

  // Confidence scales with fist tightness
  return Math.min(1, fistScore / 80);
}

/**
 * NO — index + middle fingers extended, rest folded
 * Finger pattern: thumb✗ index✓ middle✓ ring✗ pinky✗
 */
function classifyNo(geometry) {
  const { fingers } = geometry;

  // Index and middle must be up
  if (!fingers.index || !fingers.middle) return 0;

  // Ring and pinky must be folded
  if (fingers.ring || fingers.pinky) return 0;

  // Thumb can be either — in ASL NO the thumb is often alongside
  // Slight penalty if thumb is very extended (looks more like a different sign)
  const thumbPenalty = fingers.thumb ? 0.15 : 0;

  return 1.0 - thumbPenalty;
}

/**
 * SORRY — thumb extended upward, other 4 fingers folded (thumbs up / A-shape)
 * Finger pattern: thumb✓ index✗ middle✗ ring✗ pinky✗
 * Fist score must be medium-high (fingers are curled)
 */
function classifySorry(geometry) {
  const { fingers, fistScore } = geometry;

  // Thumb MUST be extended
  if (!fingers.thumb) return 0;

  // All other fingers must be folded
  const othersFolded =
    !fingers.index &&
    !fingers.middle &&
    !fingers.ring &&
    !fingers.pinky;

  if (!othersFolded) return 0;

  // Fingers should be curled (not accidentally flat)
  if (fistScore < 40) return 0;

  return Math.min(1, fistScore / 70);
}

/**
 * I LOVE YOU — thumb + index + pinky extended, middle + ring folded
 * Finger pattern: thumb✓ index✓ middle✗ ring✗ pinky✓
 * This is the ILY handshape — very unique, easy to detect
 */
function classifyILoveYou(geometry) {
  const { fingers } = geometry;

  // Thumb, index, pinky must be up
  if (!fingers.thumb || !fingers.index || !fingers.pinky) return 0;

  // Middle and ring must be folded
  if (fingers.middle || fingers.ring) return 0;

  return 1.0;
}

// ── Main classifier ─────────────────────────────────────────────────

/**
 * Runs all 5 classifiers and returns the best match.
 *
 * @param   {object} geometry - output from getHandGeometry()
 * @returns {object|null}     - { word, confidence } or null if no match
 *
 * Example return: { word: "HELLO", confidence: 0.93 }
 */
export function classifyStaticSign(geometry) {
  if (!geometry) return null;

  const candidates = [
    { word: "HELLO",       score: classifyHello(geometry)     },
    { word: "YES",         score: classifyYes(geometry)       },
    { word: "NO",          score: classifyNo(geometry)        },
    { word: "SORRY",       score: classifySorry(geometry)     },
    { word: "I LOVE YOU",  score: classifyILoveYou(geometry)  },
  ];

  // Find the highest confidence match
  const best = candidates.reduce(
    (max, c) => (c.score > max.score ? c : max),
    { word: null, score: 0 }
  );

  // Only return if it passes the threshold
  if (best.score >= CONFIDENCE_THRESHOLD) {
    return { word: best.word, confidence: best.score };
  }

  return null;
}

export { CONFIDENCE_THRESHOLD };