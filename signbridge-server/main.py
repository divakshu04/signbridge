import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import asyncio
import numpy as np
import traceback
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "signbridge_model.keras")

with open(os.path.join(BASE_DIR, "model_config.json")) as f:
    config = json.load(f)
with open(os.path.join(BASE_DIR, "label_map.json")) as f:
    IDX_TO_SIGN = {int(k): v for k, v in json.load(f).items()}

SIGNS        = config["signs"]
SEQUENCE_LEN = config["sequence_length"]
FEATURES     = config["features"]
SIGN_TO_IDX  = {v: k for k, v in IDX_TO_SIGN.items()}

# ─────────────────────────────────────────────────────────────────────
# Signaling state
# ─────────────────────────────────────────────────────────────────────
rooms: dict[str, dict] = {}

# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────
_model       = None
_model_error = None

def load_model_now():
    global _model, _model_error
    if not os.path.exists(MODEL_PATH):
        _model_error = f"Not found: {MODEL_PATH}"; print(f"✗ {_model_error}"); return
    size = os.path.getsize(MODEL_PATH)
    if size < 500_000:
        _model_error = f"Too small ({size}B) — LFS pointer?"; print(f"✗ {_model_error}"); return
    try:
        import tensorflow as tf
        print(f"[model] TF {tf.__version__}, loading…")
        _model = tf.keras.models.load_model(MODEL_PATH)
        _model.predict(np.zeros((1, SEQUENCE_LEN, FEATURES), dtype=np.float32), verbose=0)
        print("✓ Model ready")
    except MemoryError:
        _model_error = "OOM — upgrade Render plan"
    except Exception as e:
        _model_error = str(e); traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────
# Groq
# ─────────────────────────────────────────────────────────────────────
groq_client = None
try:
    from groq import Groq
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("✓ Groq ready")
except Exception as e:
    print(f"✗ Groq: {e}")

# ─────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_now()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "SignBridge Online",
        "model_loaded": _model is not None,
        "model_error": _model_error,
        "groq_ready": groq_client is not None,
        "active_rooms": list(rooms.keys()),
    }

# ─────────────────────────────────────────────────────────────────────
# Helper: safe send
# ─────────────────────────────────────────────────────────────────────
async def safe_send(ws: WebSocket, data: dict):
    try:
        await ws.send_text(json.dumps(data))
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────
# Signaling 
@app.websocket("/ws/signal/{room}/{role}")
async def signaling(ws: WebSocket, room: str, role: str):
    if role not in ("host", "guest"):
        await ws.close(code=4000); return

    await ws.accept()
    print(f"[signal] {role} connected  room={room}")

    # Register in room
    if room not in rooms:
        rooms[room] = {"host": None, "guest": None}
    rooms[room][role] = ws

    other_role = "guest" if role == "host" else "host"

    try:
        if role == "host":
            if rooms[room]["guest"]:
                await safe_send(rooms[room]["guest"], {"type": "ready"})
                await safe_send(ws, {"type": "guest_joined"})

        elif role == "guest":
            if rooms[room]["host"]:
                await safe_send(ws, {"type": "ready"})
                await safe_send(rooms[room]["host"], {"type": "guest_joined"})
            else:
                await safe_send(ws, {"type": "wait_for_host"})

        # ── Message relay loop ─────────────────────────────────────
        while True:
            try:
                raw  = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive on Render
                await safe_send(ws, {"type": "ping"})
                continue

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if msg.get("type") == "pong":
                continue

            # Forward everything else to the other peer
            other_ws = rooms.get(room, {}).get(other_role)
            if other_ws:
                await safe_send(other_ws, msg)
            else:
                # Other peer not connected yet
                await safe_send(ws, {"type": "error", "message": "Other peer not connected"})

    except WebSocketDisconnect:
        print(f"[signal] {role} disconnected  room={room}")
    except Exception as e:
        print(f"[signal] Error ({role} in {room}): {e}")
    finally:
        if room in rooms:
            rooms[room][role] = None
            # Notify other peer that this one left
            other_ws = rooms[room].get(other_role)
            if other_ws:
                await safe_send(other_ws, {"type": "peer_left"})
            # Remove room if empty
            if not rooms[room]["host"] and not rooms[room]["guest"]:
                del rooms[room]
                print(f"[signal] Room {room} closed")

# ─────────────────────────────────────────────────────────────────────
# Sign detection 
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
    if not pose or len(pose) < 17: return None
    try:
        ls, rs = pose[11], pose[12]; lw, rw = pose[15], pose[16]
        sw = abs(ls["x"] - rs["x"])
        if sw < 0.01: return None
        return {"wrist_y": (min(lw["y"], rw["y"]) - (ls["y"]+rs["y"])/2) / sw}
    except Exception: return None

def correct_probs(probs, wi):
    if wi is None: return probs
    c = probs.copy(); wy = wi["wrist_y"]
    def b(s,f):
        i=SIGN_TO_IDX.get(s,-1)
        if i>=0: c[i]=min(1.0,c[i]*f)
    def r(s,f):
        i=SIGN_TO_IDX.get(s,-1)
        if i>=0: c[i]*=f
    if wy<-0.55: b("drink",1.4);r("home",0.6)
    elif -0.55<=wy<-0.25: b("home",1.5);r("drink",0.6)
    if -0.35<=wy<0.2:
        f=SIGN_TO_IDX.get("finish",-1);p=SIGN_TO_IDX.get("please",-1)
        if f>=0 and p>=0 and c[p]>c[f] and c[p]-c[f]<0.15: b("finish",1.2)
    if wy<-0.50: b("look",1.3);r("go",0.7)
    elif wy>-0.35: b("go",1.3);r("look",0.7)
    if -0.55<wy<-0.35: b("mom",1.3)
    elif -0.70<wy<=-0.55: b("food",1.2);b("taste",1.2);b("bird",1.1);r("mom",0.8)
    if wy<-0.85: b("dad",1.3);b("hello",1.1);r("man",0.8)
    if -0.20<wy<0.30: b("time",1.2)
    if wy>0.10: b("dog",1.5)
    else: r("dog",0.7)
    t=c.sum(); return c/t if t>0 else c

CONF_THRESHOLD=0.72; CONF_GAP=0.10; VOTE_WINDOW=10
VOTE_RATIO=0.55; COOLDOWN=40; HAND_LOSS_TOLERANCE=3

def make_predictor():
    fb=deque(maxlen=SEQUENCE_LEN); vb=deque(maxlen=VOTE_WINDOW)
    lw=None; cd=0; hl=0

    def reset():
        nonlocal lw,cd,hl
        fb.clear();vb.clear();lw=None;cd=0;hl=0

    def accept(word,conf):
        nonlocal lw,cd
        lw=word;cd=COOLDOWN;vb.clear()
        recent=list(fb)[-(SEQUENCE_LEN//2):]
        fb.clear();fb.extend(recent)
        print(f"✓ {word} ({conf:.0%})")
        return {"word":word,"confidence":round(conf,3),"status":"accepted"}

    def predict(pose,left_hand,right_hand,context=""):
        nonlocal lw,cd,hl
        has=bool(left_hand) or bool(right_hand)
        if not has:
            hl+=1
            if hl<HAND_LOSS_TOLERANCE:
                fb.append(np.zeros(FEATURES,dtype=np.float32))
                return {"word":None,"status":"hand_lost","buffer":len(fb),"needed":SEQUENCE_LEN}
            reset();return {"word":None,"status":"no_hand"}
        hl=0
        if cd>0: cd-=1
        fb.append(extract_frame(pose,left_hand,right_hand))
        if len(fb)<SEQUENCE_LEN:
            return {"word":None,"status":"buffering","buffer":len(fb),"needed":SEQUENCE_LEN}
        if _model is None:
            return {"word":None,"status":"no_model","error":_model_error or "not loaded"}
        seq=np.clip(np.array(list(fb),dtype=np.float32),-3.0,3.0)[np.newaxis]

        probs=_model.predict(seq,verbose=0)[0]
        
        probs=correct_probs(probs,get_wrist_info(pose))
        ti=np.argsort(probs)[-5:][::-1]
        ts=[IDX_TO_SIGN[int(i)] for i in ti]; tc=[float(probs[i]) for i in ti]
        word=ts[0];conf=tc[0];gap=tc[0]-tc[1]
        if conf<CONF_THRESHOLD:
            vb.append(None);return {"word":None,"top_word":word,"confidence":round(conf,3),"status":"low_confidence"}
        if gap<CONF_GAP:
            vb.append(None);return {"word":None,"top_word":word,"confidence":round(conf,3),"status":"ambiguous","gap":round(gap,3)}
        vb.append(word);votes=list(vb);tv=sum(1 for v in votes if v==word);ratio=tv/len(votes)
        if ratio<VOTE_RATIO:
            return {"word":None,"top_word":word,"confidence":round(conf,3),"status":"pending","votes":f"{tv}/{len(votes)}"}
        if word==lw and cd>0:
            return {"word":None,"top_word":word,"status":"cooldown"}
        return accept(word,conf)

    return predict

@app.websocket("/ws/signs")
async def websocket_signs(ws: WebSocket):
    await ws.accept()
    predict = make_predictor()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            result = predict(data.get("pose",[]),data.get("left_hand",[]),
                             data.get("right_hand",[]),data.get("context",""))
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect: pass
    except Exception as e:
        print(f"Signs WS error: {e}")
        try: await ws.close()
        except: pass

# ─────────────────────────────────────────────────────────────────────
# Groq /suggest
# ─────────────────────────────────────────────────────────────────────
GENERAL={
    "hello":"Hello there","bye":"Goodbye, see you","yes":"Yes, I agree",
    "no":"No, I do not think so","please":"Yes please","thankyou":"Thank you so much",
    "happy":"I am feeling happy","sad":"I am feeling sad","sick":"I am not feeling well",
    "hungry":"I am hungry","sleepy":"I am feeling sleepy","sleep":"I want to sleep now",
    "drink":"I want something to drink","go":"I need to go","look":"Look at that",
    "think":"I think so","finish":"I am done","taste":"Can I taste that",
    "mom":"I want my mom","dad":"I want my dad","girl":"That is a girl",
    "boy":"That is a boy","man":"That man over there","time":"What time is it",
    "home":"I want to go home","water":"I need some water","food":"I want some food",
    "dog":"I want to see the dog","cat":"I want to see the cat","bird":"Look at that bird",
}

@app.post("/suggest")
async def suggest_sentence(request: Request):
    if not groq_client: return {"sentence":"","error":"Groq not configured"}
    body={}; sign_word=""
    try:
        body=await request.json(); sign_word=body.get("sign_word","").strip().lower()
        history=body.get("history",[])
        if not sign_word: return {"sentence":"","error":"No sign_word"}
        fallback=GENERAL.get(sign_word,sign_word.capitalize())
        recent_them=next((m.get("text","").strip() for m in reversed(history) if m.get("sender")=="them"),None)
        if not recent_them: return {"sentence":fallback}
        prompt=f"""Other person said: "{recent_them}"\nMy sign meaning: "{fallback}"\n\nWrite ONE short natural reply (3-8 words). Output only the reply.\n\nExamples:\nOther: "Hello, how are you?" | Sign: "Hello there" → I'm good, thanks.\nOther: "Are you hungry?" | Sign: "I am hungry" → Yes, I'm really hungry."""
        resp=groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":"Write short replies for a sign language user. Output only the reply."},
                      {"role":"user","content":prompt}],
            max_tokens=30,temperature=0.15)
        sentence=resp.choices[0].message.content.strip().strip('"\'`').split("\n")[0].strip()
        return {"sentence":" ".join(sentence.split()) or fallback}
    except Exception as e:
        print(f"Groq error: {e}")
        return {"sentence":GENERAL.get(sign_word,""),"error":str(e)}