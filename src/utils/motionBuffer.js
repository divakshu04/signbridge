/**
 * motionBuffer.js
 *
 * Maintains a sliding window of recent hand geometry frames
 * and extracts movement patterns from them.
 *
 * Usage:
 *   const buffer = createMotionBuffer();
 *   buffer.push(geometry);           // call every frame
 *   const motion = buffer.analyze(); // get current motion patterns
 */

const BUFFER_SIZE = 28; // ~1 second at ~30fps

export function createMotionBuffer() {
  let frames = []; // array of geometry snapshots

  return {
    // ── Push a new frame ─────────────────────────────────────────
    push(geometry) {
      if (!geometry) return;
      frames.push({
        wrist:      { ...geometry.wrist },
        palmCenter: { ...geometry.palmCenter },
        fingers:    { ...geometry.fingers },
        fistScore:  geometry.fistScore,
        tips:       {
          index:  { ...geometry.tips.index },
          middle: { ...geometry.tips.middle },
          thumb:  { ...geometry.tips.thumb },
          pinky:  { ...geometry.tips.pinky },
        },
        ts: Date.now(),
      });

      // Keep only the last BUFFER_SIZE frames
      if (frames.length > BUFFER_SIZE) {
        frames.shift();
      }
    },

    // ── Clear the buffer ─────────────────────────────────────────
    clear() {
      frames = [];
    },

    // ── How many frames we have ──────────────────────────────────
    size() {
      return frames.length;
    },

    // ── Analyze current buffer and return motion patterns ────────
    analyze() {
      if (frames.length < 10) return null;

      const first = frames[0];
      const last  = frames[frames.length - 1];
      const mid   = frames[Math.floor(frames.length / 2)];

      // ── Total displacement (start → end) ────────────────────────
      const deltaX = last.wrist.x - first.wrist.x;
      const deltaY = last.wrist.y - first.wrist.y;

      // ── Movement magnitude ──────────────────────────────────────
      const totalMovement = Math.sqrt(deltaX ** 2 + deltaY ** 2);

      // ── Primary direction ───────────────────────────────────────
      // In MediaPipe coords: Y increases downward, X increases rightward
      const movingUp    = deltaY < -0.06 && Math.abs(deltaY) > Math.abs(deltaX);
      const movingDown  = deltaY >  0.06 && Math.abs(deltaY) > Math.abs(deltaX);
      const movingLeft  = deltaX < -0.06 && Math.abs(deltaX) > Math.abs(deltaY);
      const movingRight = deltaX >  0.06 && Math.abs(deltaX) > Math.abs(deltaY);

      // ── Speed (avg movement per frame) ─────────────────────────
      let totalDist = 0;
      for (let i = 1; i < frames.length; i++) {
        const dx = frames[i].wrist.x - frames[i - 1].wrist.x;
        const dy = frames[i].wrist.y - frames[i - 1].wrist.y;
        totalDist += Math.sqrt(dx ** 2 + dy ** 2);
      }
      const avgSpeed = totalDist / frames.length;

      // ── Oscillation detection (tapping / back-and-forth) ───────
      // Count how many times the Y direction reverses
      let yReversals = 0;
      let lastYDir   = 0;
      for (let i = 1; i < frames.length; i++) {
        const dy  = frames[i].wrist.y - frames[i - 1].wrist.y;
        const dir = dy > 0.005 ? 1 : dy < -0.005 ? -1 : 0;
        if (dir !== 0 && dir !== lastYDir && lastYDir !== 0) yReversals++;
        if (dir !== 0) lastYDir = dir;
      }

      let xReversals = 0;
      let lastXDir   = 0;
      for (let i = 1; i < frames.length; i++) {
        const dx  = frames[i].wrist.x - frames[i - 1].wrist.x;
        const dir = dx > 0.005 ? 1 : dx < -0.005 ? -1 : 0;
        if (dir !== 0 && dir !== lastXDir && lastXDir !== 0) xReversals++;
        if (dir !== 0) lastXDir = dir;
      }

      const isTapping   = yReversals >= 2; // up-down repeating
      const isOscillating = xReversals >= 2 || yReversals >= 2;

      // ── Circular motion detection ───────────────────────────────
      // Sample 4 quadrant positions across the buffer and check
      // if the path goes through all 4 quadrants relative to center
      const centerX = frames.reduce((s, f) => s + f.wrist.x, 0) / frames.length;
      const centerY = frames.reduce((s, f) => s + f.wrist.y, 0) / frames.length;

      const quadrants = new Set();
      for (const f of frames) {
        const qx = f.wrist.x > centerX ? "R" : "L";
        const qy = f.wrist.y > centerY ? "D" : "U";
        quadrants.add(qx + qy);
      }
      const isCircular = quadrants.size >= 3 && totalMovement > 0.04;

      // ── Average finger state across buffer ──────────────────────
      const avgFistScore = frames.reduce((s, f) => s + f.fistScore, 0) / frames.length;

      const avgFingers = {
        thumb:  frames.filter(f => f.fingers.thumb).length  / frames.length > 0.6,
        index:  frames.filter(f => f.fingers.index).length  / frames.length > 0.6,
        middle: frames.filter(f => f.fingers.middle).length / frames.length > 0.6,
        ring:   frames.filter(f => f.fingers.ring).length   / frames.length > 0.6,
        pinky:  frames.filter(f => f.fingers.pinky).length  / frames.length > 0.6,
      };

      // ── Palm Y position (normalized) ─────────────────────────────
      // Used to check if hand is near chest area (~0.5-0.7 of frame)
      const avgPalmY = frames.reduce((s, f) => s + f.palmCenter.y, 0) / frames.length;

      // ── Wrist Y at start and end ─────────────────────────────────
      const wristYStart = first.wrist.y;
      const wristYEnd   = last.wrist.y;

      return {
        // Raw deltas
        deltaX,
        deltaY,
        totalMovement,
        avgSpeed,

        // Direction booleans
        movingUp,
        movingDown,
        movingLeft,
        movingRight,

        // Pattern booleans
        isTapping,
        isOscillating,
        isCircular,

        // Hand state averages
        avgFistScore,
        avgFingers,
        avgPalmY,

        // Wrist positions
        wristYStart,
        wristYEnd,

        // Raw frames (for advanced checks)
        frames,
      };
    },
  };
}