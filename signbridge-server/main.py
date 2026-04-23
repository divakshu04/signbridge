"""
SignBridge Server — Clean rewrite
==================================
Key behaviours:
  - Instant reset when hand removed
  - Single candidate in filter → accept after short hold (ignores model %)
  - Multiple candidates → model picks with confidence vote
  - ~2-3 second window
"""

import json
import numpy as np
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Load model ────────────────────────────────────────────────────────
print("Loading model...")
try:
    import tensorflow as tf
    model = tf.keras.models.load_model("signbridge_model.keras")
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Model load failed: {e}")
    model = None

with open("model_config.json") as f:
    config = json.load(f)
with open("label_map.json") as f:
    IDX_TO_SIGN = {int(k): v for k, v in json.load(f).items()}

SIGNS        = config["signs"]
SEQUENCE_LEN = config["sequence_length"]   # 30
FEATURES     = config["features"]           # 258
print(f"✓ {len(SIGNS)} signs ready")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "SignBridge running", "signs": len(SIGNS)}

# ── Raw frame extraction ─────────────────────────────────────────────
def extract_frame_raw(pose, left_hand, right_hand):
    pose_arr = np.zeros(132, dtype=np.float32)
    if pose:
        xyz = np.array([[p["x"], p["y"], p.get("z", 0)] for p in pose]).flatten()
        pose_arr[:min(len(xyz), pose_arr.shape[0])] = xyz[:pose_arr.shape[0]]

    lh_arr = np.zeros(63, dtype=np.float32)
    if left_hand:
        xyz = np.array([[p["x"], p["y"], p.get("z", 0)] for p in left_hand]).flatten()
        lh_arr[:min(len(xyz), lh_arr.shape[0])] = xyz[:lh_arr.shape[0]]

    rh_arr = np.zeros(63, dtype=np.float32)
    if right_hand:
        xyz = np.array([[p["x"], p["y"], p.get("z", 0)] for p in right_hand]).flatten()
        rh_arr[:min(len(xyz), rh_arr.shape[0])] = xyz[:rh_arr.shape[0]]

    frame = np.concatenate([pose_arr, lh_arr, rh_arr])
    frame = np.nan_to_num(frame, nan=0.0)
    frame = np.clip(frame, -3.0, 3.0)
    return frame, None


# ── Wrist position from pose ──────────────────────────────────────────
def get_wrist_info(pose):
    """
    Returns normalized wrist Y relative to shoulders.
    Used only for lightweight post-model correction.

    Returns dict with:
      wrist_y       : normalized Y (negative = above shoulders)
      shoulder_y    : raw shoulder Y midpoint
      shoulder_width: raw shoulder width (scale reference)
    """
    if not pose or len(pose) < 17:
        return None

    try:
        ls = pose[11]  # left shoulder
        rs = pose[12]  # right shoulder
        lw = pose[15]  # left wrist
        rw = pose[16]  # right wrist

        shoulder_y     = (ls["y"] + rs["y"]) / 2
        shoulder_width = abs(ls["x"] - rs["x"])

        if shoulder_width < 0.01:
            return None

        # Use whichever wrist is higher (more likely active)
        wrist_y = min(lw["y"], rw["y"])

        # Normalize relative to shoulders
        norm_y = (wrist_y - shoulder_y) / shoulder_width

        return {
            "wrist_y":        norm_y,
            "shoulder_y":     shoulder_y,
            "shoulder_width": shoulder_width,
        }
    except Exception:
        return None


# ── Post-model confusion correction ──────────────────────────────────
def correct_probs(probs, sign_to_idx, wrist_info):
    """
    Lightweight targeted correction for known confused pairs.
    Only nudges probabilities for specific signs based on wrist height.
    Does NOT affect signs that are already working well.

    Corrections applied:
      home vs drink:
        - home: cheek/chin level (wrist_y between -0.8 and -0.3)
        - drink: mouth level, slightly higher (wrist_y < -0.5)

      finish vs please:
        - finish: has outward movement (harder to detect here, boost if
          please is winning but finish is close)
        - please: circular at chest

      sad vs sleepy vs sleep:
        - sad: face level, both hands (wrist_y < -0.6)
        - sleepy/sleep: face level dropping

      go vs look:
        - look: closer to face (wrist_y < -0.5)
        - go: neutral space (wrist_y > -0.4)

      man vs dad:
        - dad: forehead (wrist_y < -0.9)
        - man: forehead then chest movement

      mom vs food vs taste vs bird:
        - food/taste/bird: at mouth (wrist_y between -0.7 and -0.4)
        - mom: slightly lower at chin
    """
    if wrist_info is None:
        return probs

    corrected = probs.copy()
    wy = wrist_info["wrist_y"]

    def idx(sign):
        return sign_to_idx.get(sign, -1)

    def boost(sign, factor):
        i = idx(sign)
        if i >= 0:
            corrected[i] = min(1.0, corrected[i] * factor)

    def reduce(sign, factor):
        i = idx(sign)
        if i >= 0:
            corrected[i] = corrected[i] * factor

    # ── home vs drink ─────────────────────────────────────────────
    # home is at cheek (lower), drink is at mouth (higher/closer to face)
    if wy < -0.55:
        # Hand is near face/mouth — drink more likely than home
        boost("drink", 1.4)
        reduce("home", 0.6)
    elif -0.55 <= wy < -0.25:
        # Hand is at cheek/chin — home more likely
        boost("home", 1.5)
        reduce("drink", 0.6)

    # ── finish vs please ──────────────────────────────────────────
    # Both at chest level — hard to separate without movement
    # please is circular, finish is flip outward
    # Slight boost to finish when at neutral/chest (it's often too low score)
    if -0.35 <= wy < 0.2:
        i_finish = idx("finish")
        i_please = idx("please")
        if i_finish >= 0 and i_please >= 0:
            if corrected[i_please] > corrected[i_finish]:
                # please is winning — check if finish is close behind
                gap = corrected[i_please] - corrected[i_finish]
                if gap < 0.15:
                    # Too close to call — slight finish boost
                    boost("finish", 1.2)

    # ── go vs look ────────────────────────────────────────────────
    # look is near face, go is in neutral space in front
    if wy < -0.50:
        boost("look", 1.3)
        reduce("go", 0.7)
    elif wy > -0.35:
        boost("go", 1.3)
        reduce("look", 0.7)

    # ── mom vs food vs taste vs bird ─────────────────────────────
    # All near mouth — mom is specifically at chin (slightly lower)
    if -0.55 < wy < -0.35:
        # Chin level — mom territory
        boost("mom", 1.3)
    elif -0.70 < wy <= -0.55:
        # Mouth level — food/taste/bird territory
        boost("food",  1.2)
        boost("taste", 1.2)
        boost("bird",  1.1)
        reduce("mom",  0.8)

    # ── dad vs man vs boy vs hello ────────────────────────────────
    # dad and hello are at forehead, man transitions forehead→chest
    if wy < -0.85:
        # Very high — forehead level
        boost("dad",   1.3)
        boost("hello", 1.2)
        reduce("man",  0.8)

    # ── sad vs sleepy vs sleep ────────────────────────────────────
    # All face level downward — hard to separate
    # No strong position signal here — leave to model

    # ── cat vs home ───────────────────────────────────────────────
    # cat is at cheek pulling outward, home is chin tapping
    if -0.60 < wy < -0.30:
        i_cat  = idx("cat")
        i_home = idx("home")
        if i_cat >= 0 and i_home >= 0:
            if abs(corrected[i_cat] - corrected[i_home]) < 0.12:
                # Very close — no strong correction, trust model
                pass

    # ── time ─────────────────────────────────────────────────────
    # time requires tapping wrist — usually at neutral/low level
    if -0.20 < wy < 0.30:
        boost("time", 1.2)

    # ── dog ──────────────────────────────────────────────────────
    # dog is at side/thigh level — very low position
    if wy > 0.10:
        boost("dog", 1.5)
    else:
        reduce("dog", 0.7)

    # Renormalize so probabilities sum to 1
    total = corrected.sum()
    if total > 0:
        corrected = corrected / total

    return corrected


# ── Settings ──────────────────────────────────────────────────────────
CONF_THRESHOLD      = 0.72
CONF_GAP            = 0.10
VOTE_WINDOW         = 10
VOTE_RATIO          = 0.55
COOLDOWN            = 40
HAND_LOSS_TOLERANCE = 3

# Build sign→idx map for correction function
SIGN_TO_IDX = {v: k for k, v in IDX_TO_SIGN.items()}


# ── Context-aware sign filtering ─────────────────────────────────────
def get_contextual_signs(conversation_context):
    """
    NOTE: This function no longer modifies sign detection.
    Context filtering was causing false positives.
    
    Now we use RAW model predictions for all signs equally.
    Context is ONLY used by Groq for sentence generation.
    """
    return {}  # Return empty - no filtering, raw model only


def apply_context_filter(probs, sign_to_idx, contextual_boosts):
    """
    Applies soft context-based filtering to prediction probabilities.
    Uses contextual boosts to guide predictions without suppressing model accuracy.
    """
    filtered = probs.copy()
    
    for sign, boost_factor in contextual_boosts.items():
        idx = sign_to_idx.get(sign, -1)
        if idx >= 0:
            filtered[idx] = min(1.0, filtered[idx] * boost_factor)
    
    # Renormalize
    total = filtered.sum()
    if total > 0:
        filtered = filtered / total
    
    return filtered


# ── Predictor ─────────────────────────────────────────────────────────
def make_predictor():
    frame_buf       = deque(maxlen=SEQUENCE_LEN)
    vote_buf        = deque(maxlen=VOTE_WINDOW)
    last_word       = None
    cooldown        = 0
    hand_loss_count = 0
    context_history = deque(maxlen=5)  # Track recent context

    def reset():
        nonlocal last_word, cooldown, hand_loss_count
        frame_buf.clear()
        vote_buf.clear()
        last_word       = None
        cooldown        = 0
        hand_loss_count = 0

    def accept(word, conf):
        nonlocal last_word, cooldown
        last_word = word
        cooldown  = COOLDOWN
        vote_buf.clear()
        recent = list(frame_buf)[-(SEQUENCE_LEN // 2):]
        frame_buf.clear()
        frame_buf.extend(recent)
        print(f"✓ {word} ({conf:.0%})")
        return {"word": word, "confidence": round(conf, 3), "status": "accepted"}

    def predict(pose, left_hand, right_hand, conversation_context=""):
        nonlocal last_word, cooldown, hand_loss_count

        # Track conversation context for filtering
        if conversation_context.strip():
            context_history.append(conversation_context)

        has_hand = len(left_hand) > 0 or len(right_hand) > 0

        if not has_hand:
            hand_loss_count += 1
            if hand_loss_count < HAND_LOSS_TOLERANCE:
                frame_buf.append(np.zeros(FEATURES, dtype=np.float32))
                return {
                    "word":   None,
                    "status": "hand_lost",
                    "lost_frames": hand_loss_count,
                    "buffer": len(frame_buf),
                    "needed": SEQUENCE_LEN,
                }
            reset()
            return {"word": None, "status": "no_hand"}

        hand_loss_count = 0
        if cooldown > 0:
            cooldown -= 1

        frame, _ = extract_frame_raw(pose, left_hand, right_hand)
        frame_buf.append(frame)

        if len(frame_buf) < SEQUENCE_LEN:
            return {
                "word":   None,
                "status": "buffering",
                "buffer": len(frame_buf),
                "needed": SEQUENCE_LEN,
            }

        if model is None:
            return {"word": None, "status": "no_model"}

        seq   = np.clip(np.array(list(frame_buf), dtype=np.float32),
                        -3.0, 3.0)[np.newaxis, ...]
        
        probs = model.predict(seq, verbose=0)[0]

        # ── Apply lightweight confusion correction ────────────────────
        # Re-enabled: This helps distinguish similar signs like dad/man
        wrist_info = get_wrist_info(pose)
        probs      = correct_probs(probs, SIGN_TO_IDX, wrist_info)

        # NOTE: Context filtering disabled - using raw model predictions only
        # Groq handles context for sentence generation, not sign detection

        top5_idx   = np.argsort(probs)[-5:][::-1]
        top5_signs = [IDX_TO_SIGN.get(int(i), "?") for i in top5_idx]
        top5_conf  = [float(probs[i]) for i in top5_idx]

        filtered_word = top5_signs[0]
        filtered_conf = top5_conf[0]
        conf_gap      = top5_conf[0] - top5_conf[1]

        if filtered_conf < CONF_THRESHOLD:
            vote_buf.append(None)
            return {
                "word":       None,
                "confidence": round(filtered_conf, 3),
                "top_word":   filtered_word,
                "status":     "low_confidence",
            }

        if conf_gap < CONF_GAP:
            vote_buf.append(None)
            return {
                "word":       None,
                "confidence": round(filtered_conf, 3),
                "top_word":   filtered_word,
                "status":     "ambiguous",
                "gap":        round(conf_gap, 3),
            }

        vote_buf.append(filtered_word)
        votes     = list(vote_buf)
        top_votes = sum(1 for v in votes if v == filtered_word)
        ratio     = top_votes / len(votes)

        if ratio < VOTE_RATIO:
            return {
                "word":       None,
                "confidence": round(filtered_conf, 3),
                "top_word":   filtered_word,
                "status":     "pending",
                "votes":      f"{top_votes}/{len(votes)}",
            }

        if filtered_word == last_word and cooldown > 0:
            return {
                "word":     None,
                "status":   "cooldown",
                "top_word": filtered_word,
            }

        return accept(filtered_word, filtered_conf)

    return predict


# ── WebSocket ─────────────────────────────────────────────────────────
@app.websocket("/ws/signs")
async def websocket_signs(ws: WebSocket):
    await ws.accept()
    print("\n✓ Browser connected")
    predict = make_predictor()
    try:
        while True:
            data   = json.loads(await ws.receive_text())
            # Extract context from client if provided
            conversation_context = data.get("context", "")
            result = predict(data.get("pose", []),
                             data.get("left_hand", []),
                             data.get("right_hand", []),
                             conversation_context)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print("✗ Disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try: await ws.close()
        except: pass


# ── Groq sentence suggestion endpoint ────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("✓ Groq client ready")
except Exception as e:
    groq_client = None
    print(f"✗ Groq not available: {e}")

from fastapi import Request

@app.post("/suggest")
async def suggest_sentence(request: Request):
    """
    For every sign:
    - If other person sent a message recently → Groq replies to it using the sign
    - If no recent context → returns a general fallback sentence for that sign
    """
    if not groq_client:
        return {"sentence": "", "error": "Groq not configured"}

    try:
        body      = await request.json()
        sign_word = body.get("sign_word", "").strip().lower()
        history   = body.get("history", [])

        if not sign_word:
            return {"sentence": "", "error": "No sign word"}

        # ── General fallback sentences for all 30 signs ───────────────
        # These work in any context when there is nothing to reply to
        GENERAL = {
            "hello":    "Hello there",
            "bye":      "Goodbye, see you",
            "yes":      "Yes, I agree",
            "no":       "No, I do not think so",
            "please":   "Yes please",
            "thankyou": "Thank you so much",
            "happy":    "I am feeling happy",
            "sad":      "I am feeling sad",
            "sick":     "I am not feeling well",
            "hungry":   "I am hungry",
            "sleepy":   "I am feeling sleepy",
            "sleep":    "I want to sleep now",
            "drink":    "I want something to drink",
            "go":       "I need to go",
            "look":     "Look at that",
            "think":    "I think so",
            "finish":   "I am done",
            "taste":    "Can I taste that",
            "mom":      "I want my mom",
            "dad":      "I want my dad",
            "girl":     "That is a girl",
            "boy":      "That is a boy",
            "man":      "That man over there",
            "time":     "What time is it",
            "home":     "I want to go home",
            "water":    "I need some water",
            "food":     "I want some food",
            "dog":      "I want to see the dog",
            "cat":      "I want to see the cat",
            "bird":     "Look at that bird",
        }

        # ── Check if there is a SINGLE recent message to reply to ────────────
        # Only consider the LAST message (most recent), only from the other person
        recent_them = None
        if history:
            # Check ONLY the last message in history
            last_msg = history[-1]
            if last_msg.get("sender") == "them":
                recent_them = last_msg.get("text", "").strip()

        # If no recent message from other person, return general fallback
        if not recent_them:
            fallback = GENERAL.get(sign_word, sign_word.capitalize())
            return {"sentence": fallback}

        # ── There is a recent message — use Groq to reply ─────────────
        general_fallback = GENERAL.get(sign_word, sign_word.capitalize())

        prompt = f"""Other person said: "{recent_them}"
I signed the word: "{sign_word}"
My general meaning for this sign: "{general_fallback}"

Write ONE short natural reply (3-8 words) that:
- Directly answers what they said
- Uses the meaning of my sign word
- Sounds like normal conversation

Only output the reply sentence, nothing else."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=25,
            temperature=0.2,
        )

        sentence = response.choices[0].message.content.strip()
        sentence = sentence.strip('"\'`').split("\n")[0].strip()
        sentence = " ".join(sentence.split())

        # Fallback if Groq returns empty or just the sign word
        if not sentence or sentence.lower() == sign_word:
            sentence = general_fallback

        return {"sentence": sentence}

    except Exception as e:
        print(f"Groq error: {e}")
        fallback = GENERAL.get(sign_word, sign_word.capitalize()) if 'sign_word' in dir() else ""
        return {"sentence": fallback, "error": str(e)}