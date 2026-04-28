import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import numpy as np
import traceback
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# Paths & config
# ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "signbridge_model.keras")

with open(os.path.join(BASE_DIR, "model_config.json")) as f:
    config = json.load(f)
with open(os.path.join(BASE_DIR, "label_map.json")) as f:
    IDX_TO_SIGN = {int(k): v for k, v in json.load(f).items()}

SIGNS        = config["signs"]
SEQUENCE_LEN = config["sequence_length"]   # 30
FEATURES     = config["features"]           # 258
SIGN_TO_IDX  = {v: k for k, v in IDX_TO_SIGN.items()}

# ─────────────────────────────────────────────────────────────────────
# Model — loaded ONCE at startup via lifespan
# ─────────────────────────────────────────────────────────────────────
_model       = None
_model_error = None

def load_model_now():
    """Import TF and load the model. Called once at startup."""
    global _model, _model_error

    if not os.path.exists(MODEL_PATH):
        _model_error = f"File not found: {MODEL_PATH}"
        print(f"✗ {_model_error}")
        return

    size = os.path.getsize(MODEL_PATH)
    print(f"[model] {MODEL_PATH} — {size:,} bytes")

    if size < 500_000:
        _model_error = (
            f"File only {size} bytes — looks like a Git LFS pointer, not real weights. "
            "Add 'git lfs pull' to your Render build command."
        )
        print(f"✗ {_model_error}")
        return

    try:
        print("[model] Importing TensorFlow…")
        import tensorflow as tf
        print(f"[model] TensorFlow {tf.__version__}")
        print("[model] Loading weights…")
        _model = tf.keras.models.load_model(MODEL_PATH)
        # Warm-up: first predict is slow; do it now so WebSocket users don't wait
        _model.predict(np.zeros((1, SEQUENCE_LEN, FEATURES), dtype=np.float32), verbose=0)
        print("✓ Model ready")
    except MemoryError:
        _model_error = "Out of memory — upgrade Render plan to ≥ 1 GB RAM"
        print(f"✗ {_model_error}")
    except Exception as e:
        _model_error = str(e)
        print("✗ Model load failed:")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────
# Groq client
# ─────────────────────────────────────────────────────────────────────
groq_client = None
_groq_error = None
try:
    from groq import Groq
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("✓ Groq ready")
except Exception as e:
    _groq_error = str(e)
    print(f"✗ Groq: {e}")


# ─────────────────────────────────────────────────────────────────────
# FastAPI — lifespan loads the model before any request is served
# ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("── startup: loading model ──")
    load_model_now()      # THIS is the fix — was never being called before
    print("── startup complete ──")
    yield
    print("── shutdown ──")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Health endpoints
# ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":       "SignBridge Online",
        "model_loaded": _model is not None,
        "model_error":  _model_error,
        "groq_ready":   groq_client is not None,
        "signs":        len(SIGNS),
    }

@app.get("/health")
def health():
    import sys
    exists = os.path.exists(MODEL_PATH)
    size   = os.path.getsize(MODEL_PATH) if exists else 0
    return {
        "python":               sys.version,
        "model_path":           MODEL_PATH,
        "model_file_exists":    exists,
        "model_file_bytes":     size,
        "model_is_lfs_pointer": size < 500_000,
        "model_loaded":         _model is not None,
        "model_error":          _model_error,
        "groq_api_key_set":     bool(os.environ.get("GROQ_API_KEY")),
        "groq_ready":           groq_client is not None,
        "groq_error":           _groq_error,
        "signs":                len(SIGNS),
    }


# ─────────────────────────────────────────────────────────────────────
# Frame extraction helpers
# ─────────────────────────────────────────────────────────────────────
def extract_frame(pose, left_hand, right_hand):
    def to_arr(lm, size):
        arr = np.zeros(size, dtype=np.float32)
        if lm:
            xyz = np.array([[p["x"], p["y"], p.get("z", 0)] for p in lm]).flatten()
            arr[:min(len(xyz), size)] = xyz[:size]
        return arr
    frame = np.concatenate([to_arr(pose, 132), to_arr(left_hand, 63), to_arr(right_hand, 63)])
    return np.clip(np.nan_to_num(frame, nan=0.0), -3.0, 3.0)


def get_wrist_info(pose):
    if not pose or len(pose) < 17:
        return None
    try:
        ls, rs = pose[11], pose[12]
        lw, rw = pose[15], pose[16]
        sw = abs(ls["x"] - rs["x"])
        if sw < 0.01:
            return None
        sy = (ls["y"] + rs["y"]) / 2
        return {"wrist_y": (min(lw["y"], rw["y"]) - sy) / sw}
    except Exception:
        return None


def correct_probs(probs, wrist_info):
    if wrist_info is None:
        return probs
    c  = probs.copy()
    wy = wrist_info["wrist_y"]

    def b(sign, f):
        i = SIGN_TO_IDX.get(sign, -1)
        if i >= 0: c[i] = min(1.0, c[i] * f)

    def r(sign, f):
        i = SIGN_TO_IDX.get(sign, -1)
        if i >= 0: c[i] *= f

    if wy < -0.55:            b("drink", 1.4); r("home", 0.6)
    elif -0.55 <= wy < -0.25: b("home",  1.5); r("drink", 0.6)

    if -0.35 <= wy < 0.2:
        i_f = SIGN_TO_IDX.get("finish", -1)
        i_p = SIGN_TO_IDX.get("please", -1)
        if i_f >= 0 and i_p >= 0 and c[i_p] > c[i_f] and (c[i_p] - c[i_f]) < 0.15:
            b("finish", 1.2)

    if wy < -0.50:    b("look", 1.3); r("go",  0.7)
    elif wy > -0.35:  b("go",   1.3); r("look", 0.7)

    if   -0.55 < wy < -0.35:  b("mom", 1.3)
    elif -0.70 < wy <= -0.55: b("food", 1.2); b("taste", 1.2); b("bird", 1.1); r("mom", 0.8)

    if wy < -0.85:         b("dad", 1.3); b("hello", 1.1); r("man", 0.8)
    if -0.20 < wy < 0.30:  b("time", 1.2)
    if wy > 0.10:          b("dog", 1.5)
    else:                  r("dog", 0.7)

    total = c.sum()
    return c / total if total > 0 else c


# ─────────────────────────────────────────────────────────────────────
# Per-connection predictor
# ─────────────────────────────────────────────────────────────────────
CONF_THRESHOLD      = 0.72
CONF_GAP            = 0.10
VOTE_WINDOW         = 10
VOTE_RATIO          = 0.55
COOLDOWN            = 40
HAND_LOSS_TOLERANCE = 3

def make_predictor():
    frame_buf       = deque(maxlen=SEQUENCE_LEN)
    vote_buf        = deque(maxlen=VOTE_WINDOW)
    last_word       = None
    cooldown        = 0
    hand_loss_count = 0

    def reset():
        nonlocal last_word, cooldown, hand_loss_count
        frame_buf.clear(); vote_buf.clear()
        last_word = None; cooldown = 0; hand_loss_count = 0

    def accept(word, conf):
        nonlocal last_word, cooldown
        last_word = word; cooldown = COOLDOWN
        vote_buf.clear()
        recent = list(frame_buf)[-(SEQUENCE_LEN // 2):]
        frame_buf.clear(); frame_buf.extend(recent)
        print(f"✓ {word} ({conf:.0%})")
        return {"word": word, "confidence": round(conf, 3), "status": "accepted"}

    def predict(pose, left_hand, right_hand, context=""):
        nonlocal last_word, cooldown, hand_loss_count

        has_hand = bool(left_hand) or bool(right_hand)

        if not has_hand:
            hand_loss_count += 1
            if hand_loss_count < HAND_LOSS_TOLERANCE:
                frame_buf.append(np.zeros(FEATURES, dtype=np.float32))
                return {"word": None, "status": "hand_lost",
                        "buffer": len(frame_buf), "needed": SEQUENCE_LEN}
            reset()
            return {"word": None, "status": "no_hand"}

        hand_loss_count = 0
        if cooldown > 0:
            cooldown -= 1

        frame_buf.append(extract_frame(pose, left_hand, right_hand))

        if len(frame_buf) < SEQUENCE_LEN:
            return {"word": None, "status": "buffering",
                    "buffer": len(frame_buf), "needed": SEQUENCE_LEN}

        if _model is None:
            return {"word": None, "status": "no_model",
                    "error": _model_error or "Model failed to load at startup — check /health"}

        seq   = np.clip(np.array(list(frame_buf), dtype=np.float32), -3.0, 3.0)[np.newaxis]
        probs = _model.predict(seq, verbose=0)[0]
        probs = correct_probs(probs, get_wrist_info(pose))

        top_idx  = np.argsort(probs)[-5:][::-1]
        top_sign = [IDX_TO_SIGN[int(i)] for i in top_idx]
        top_conf = [float(probs[i])     for i in top_idx]

        word = top_sign[0]
        conf = top_conf[0]
        gap  = top_conf[0] - top_conf[1]

        if conf < CONF_THRESHOLD:
            vote_buf.append(None)
            return {"word": None, "top_word": word, "confidence": round(conf, 3),
                    "status": "low_confidence"}

        if gap < CONF_GAP:
            vote_buf.append(None)
            return {"word": None, "top_word": word, "confidence": round(conf, 3),
                    "status": "ambiguous", "gap": round(gap, 3)}

        vote_buf.append(word)
        votes     = list(vote_buf)
        top_votes = sum(1 for v in votes if v == word)
        ratio     = top_votes / len(votes)

        if ratio < VOTE_RATIO:
            return {"word": None, "top_word": word, "confidence": round(conf, 3),
                    "status": "pending", "votes": f"{top_votes}/{len(votes)}"}

        if word == last_word and cooldown > 0:
            return {"word": None, "top_word": word, "status": "cooldown"}

        return accept(word, conf)

    return predict


# ─────────────────────────────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────────────────────────────
@app.websocket("/ws/signs")
async def websocket_signs(ws: WebSocket):
    await ws.accept()
    print("✓ Browser connected")
    predict = make_predictor()
    try:
        while True:
            data   = json.loads(await ws.receive_text())
            result = predict(
                data.get("pose", []),
                data.get("left_hand", []),
                data.get("right_hand", []),
                data.get("context", ""),
            )
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print("✗ Browser disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try: await ws.close()
        except: pass


# ─────────────────────────────────────────────────────────────────────
# Groq /suggest
# ─────────────────────────────────────────────────────────────────────
GENERAL = {
    "hello": "Hello there", "bye": "Goodbye, see you",
    "yes": "Yes, I agree", "no": "No, I do not think so",
    "please": "Yes please", "thankyou": "Thank you so much",
    "happy": "I am feeling happy", "sad": "I am feeling sad",
    "sick": "I am not feeling well", "hungry": "I am hungry",
    "sleepy": "I am feeling sleepy", "sleep": "I want to sleep now",
    "drink": "I want something to drink", "go": "I need to go",
    "look": "Look at that", "think": "I think so",
    "finish": "I am done", "taste": "Can I taste that",
    "mom": "I want my mom", "dad": "I want my dad",
    "girl": "That is a girl", "boy": "That is a boy",
    "man": "That man over there", "time": "What time is it",
    "home": "I want to go home", "water": "I need some water",
    "food": "I want some food", "dog": "I want to see the dog",
    "cat": "I want to see the cat", "bird": "Look at that bird",
}

@app.post("/suggest")
async def suggest_sentence(request: Request):
    if not groq_client:
        return {"sentence": "", "error": "Groq not configured — set GROQ_API_KEY env var"}

    body      = {}
    sign_word = ""
    try:
        body      = await request.json()
        sign_word = body.get("sign_word", "").strip().lower()
        history   = body.get("history", [])

        if not sign_word:
            return {"sentence": "", "error": "No sign_word provided"}

        fallback    = GENERAL.get(sign_word, sign_word.capitalize())
        recent_them = next(
            (m.get("text", "").strip() for m in reversed(history)
             if m.get("sender") == "them"),
            None
        )

        if not recent_them:
            return {"sentence": fallback}

        prompt = f"""Other person said: "{recent_them}"
My sign meaning: "{fallback}"

Write ONE short natural reply (3-8 words) that directly answers what they said and matches the sign intent. Output only the reply, no labels or explanation.

Examples:
Other: "Hello, how are you?" | Sign: "Hello there" → I'm good, thanks.
Other: "Where are you going?" | Sign: "I want my dad" → I'm going to see my dad.
Other: "Are you hungry?" | Sign: "I am hungry" → Yes, I'm really hungry."""

        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You write short conversational replies for a sign language user. Output only the reply sentence."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=30,
            temperature=0.15,
        )
        sentence = resp.choices[0].message.content.strip().strip('"\'`').split("\n")[0].strip()
        return {"sentence": " ".join(sentence.split()) or fallback}

    except Exception as e:
        print(f"Groq error: {e}")
        return {"sentence": GENERAL.get(sign_word, ""), "error": str(e)}