import { useState, useRef, useEffect, useCallback } from "react";
import StarCanvas from "../components/StarCanvas";
import { useHandDetection } from "../hooks/useHandDetection";
import styles from "./CallRoom.module.css";

// --- Icons ---
const IconMic    = () => <svg viewBox="0 0 24 24" fill="none" width="20" height="20"><rect x="9" y="2" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="1.8"/><path d="M5 10a7 7 0 0014 0M12 19v3M9 22h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>;
const IconMicOff = () => <svg viewBox="0 0 24 24" fill="none" width="20" height="20"><line x1="2" y1="2" x2="22" y2="22" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><path d="M9 9v5a3 3 0 005.16 2.08M15 9.34V4a3 3 0 00-5.94-.6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><path d="M17 16.95A7 7 0 015 10M12 19v3M9 22h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>;
const IconCam    = () => <svg viewBox="0 0 24 24" fill="none" width="20" height="20"><path d="M15 10l4.55-2.37A1 1 0 0121 8.62v6.76a1 1 0 01-1.45.9L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>;
const IconCamOff = () => <svg viewBox="0 0 24 24" fill="none" width="20" height="20"><line x1="2" y1="2" x2="22" y2="22" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><path d="M10.68 6H13a2 2 0 012 2v2.32M15 15H5a2 2 0 01-2-2V8a2 2 0 012-2M21 8.62v6.76a1 1 0 01-1.45.9L15 14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>;
const IconStt    = () => <svg viewBox="0 0 24 24" fill="none" width="20" height="20"><path d="M12 2a3 3 0 00-3 3v6a3 3 0 006 0V5a3 3 0 00-3-3z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><path d="M19 10v1a7 7 0 01-14 0v-1M12 18v4M8 22h8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>;
const IconSttOff = () => <svg viewBox="0 0 24 24" fill="none" width="20" height="20"><line x1="2" y1="2" x2="22" y2="22" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><path d="M9 9v3a3 3 0 005.12 2.12M15 9.34V6a3 3 0 00-5.94-.64" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><path d="M17 14.95A7 7 0 015 11v-1M12 18v4M8 22h8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>;
const IconPhone  = () => <svg viewBox="0 0 24 24" fill="white" width="20" height="20"><path d="M6.6 10.8c1.4 2.8 3.8 5.1 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1-9.4 0-17-7.6-17-17 0-.6.4-1 1-1h3.5c.6 0 1 .4 1 1 0 1.3.2 2.5.6 3.6.1.3 0 .7-.2 1L6.6 10.8z" fill="white"/></svg>;
const IconDots   = () => <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/></svg>;
const IconMenu   = () => <svg viewBox="0 0 24 24" fill="none" width="18" height="18"><path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>;
const IconSend   = () => <svg viewBox="0 0 24 24" fill="none" width="18" height="18"><path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>;
const IconEmoji  = () => <svg viewBox="0 0 24 24" fill="none" width="20" height="20"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.8"/><path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><circle cx="9" cy="10" r="1" fill="currentColor"/><circle cx="15" cy="10" r="1" fill="currentColor"/></svg>;
const IconMicSm  = () => <svg viewBox="0 0 24 24" fill="none" width="13" height="13"><rect x="9" y="2" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="2"/><path d="M5 10a7 7 0 0014 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>;
const IconPlus   = () => <svg viewBox="0 0 24 24" fill="none" width="18" height="18"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>;
const IconUser   = () => <svg viewBox="0 0 24 24" fill="none" width="48" height="48"><circle cx="12" cy="8" r="5" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5"/><path d="M3 21c0-5 4-9 9-9s9 4 9 9" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" strokeLinecap="round"/></svg>;

// --- WebRTC Config ---
const ICE = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
    { urls: "turn:openrelay.metered.ca:80",                username: "openrelayproject", credential: "openrelayproject" },
    { urls: "turn:openrelay.metered.ca:443",               username: "openrelayproject", credential: "openrelayproject" },
    { urls: "turn:openrelay.metered.ca:443?transport=tcp", username: "openrelayproject", credential: "openrelayproject" },
  ],
};

const API    = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const WS_API = API.replace(/^https/, "wss").replace(/^http/, "ws");

export default function CallRoom({ roomCode, isHost, onLeave }) {
  // --- States ---
  const [localIsHost, setLocalIsHost] = useState(isHost); // The actual role assigned by server
  const [micOn,      setMicOn]      = useState(false);
  const [camOn,      setCamOn]      = useState(false);
  const [sttOn,      setSttOn]      = useState(false);
  const [peerStatus, setPeerStatus] = useState("connecting");
  const [statusMsg,  setStatusMsg]  = useState("Connecting to server…");
  const [messages,   setMessages]   = useState([]);
  const [inputText,  setInputText]  = useState("");
  const [isThinking, setIsThinking] = useState(false);

  // --- Refs ---
  const localVideoRef  = useRef(null);
  const remoteVideoRef = useRef(null);
  const localStream    = useRef(null);
  const pc             = useRef(null);
  const sig            = useRef(null);
  const chatEndRef     = useRef(null);
  const messagesRef    = useRef([]);
  const inputSourceRef = useRef("manual");
  const lastRepliedRef = useRef(null);
  const recognitionRef = useRef(null);
  const dead           = useRef(false);
  const makingOffer    = useRef(false);

  // --- Helpers ---
  const addMsg = useCallback((text, sender, type = "text") => {
    const msg = { id: Date.now() + Math.random(), text, sender, type,
                  time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) };
    setMessages(prev => { const u = [...prev, msg]; messagesRef.current = u; return u; });
  }, []);

  const sendSig = useCallback((obj) => {
    if (sig.current?.readyState === WebSocket.OPEN) sig.current.send(JSON.stringify(obj));
  }, []);

  const createPC = useCallback(() => {
    if (pc.current) { pc.current.close(); pc.current = null; }
    const conn = new RTCPeerConnection(ICE);
    localStream.current?.getTracks().forEach(t => conn.addTrack(t, localStream.current));

    conn.ontrack = (e) => {
      if (dead.current) return;
      if (remoteVideoRef.current) { 
        remoteVideoRef.current.srcObject = e.streams[0]; 
        remoteVideoRef.current.play().catch(() => {}); 
      }
      setPeerStatus("connected"); 
      setStatusMsg("Live");
      addMsg("Connected! You are now in the same room.", "system");
    };

    conn.onicecandidate = (e) => { 
      if (e.candidate) sendSig({ type: "ice", candidate: e.candidate }); 
    };

    conn.onconnectionstatechange = () => {
      if (dead.current) return;
      const s = conn.connectionState;
      if (s === "failed") { 
        setPeerStatus("error"); 
        setStatusMsg("Connection failed"); 
        addMsg("WebRTC failed. Please refresh.", "system"); 
      }
      if (s === "disconnected") { 
        setPeerStatus("waiting"); 
        setStatusMsg("Other person disconnected"); 
        if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null; 
        addMsg("The other person left.", "system"); 
      }
    };

    pc.current = conn; 
    return conn;
  }, [addMsg, sendSig]);

  const startOffer = useCallback(async () => {
    if (makingOffer.current) return;
    makingOffer.current = true;
    try {
      const conn = createPC();
      const offer = await conn.createOffer();
      await conn.setLocalDescription(offer);
      sendSig({ type: "offer", sdp: conn.localDescription });
    } catch (e) { 
      console.error("offer failed:", e); 
      makingOffer.current = false; 
    }
  }, [createPC, sendSig]);

  // --- Core Lifecycle ---
  useEffect(() => {
    dead.current = false;
    async function init() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        stream.getVideoTracks().forEach(t => t.enabled = false);
        stream.getAudioTracks().forEach(t  => t.enabled = false);
        localStream.current = stream;
        if (localVideoRef.current) localVideoRef.current.srcObject = stream;
      } catch {
        setPeerStatus("error"); 
        setStatusMsg("Camera/mic denied"); 
        return;
      }

      const safeRoom = roomCode.replace(/[^a-zA-Z0-9-]/g, "");
      const role     = isHost ? "host" : "guest";
      const url      = `${WS_API}/ws/signal/${safeRoom}/${role}`;
      
      const ws = new WebSocket(url);
      sig.current = ws;

      ws.onmessage = async (e) => {
        if (dead.current) return;
        let msg; try { msg = JSON.parse(e.data); } catch { return; }
        
        switch (msg.type) {
          case "assigned_role":
            const actualHost = (msg.role === "host");
            setLocalIsHost(actualHost);
            setPeerStatus("waiting");
            setStatusMsg(actualHost ? `Waiting for guest — code: ${roomCode}` : "Looking for host…");
            addMsg(`You joined as ${msg.role}.`, "system");
            break;

          case "ready":
            // Guests always initiate the offer to avoid collisions
            if (!localIsHost) {
              await startOffer();
            }
            break;

          case "offer":
            if (localIsHost) {
              const conn = createPC();
              await conn.setRemoteDescription(new RTCSessionDescription(msg.sdp));
              const answer = await conn.createAnswer();
              await conn.setLocalDescription(answer);
              sendSig({ type: "answer", sdp: conn.localDescription });
            }
            break;

          case "answer":
            if (!localIsHost && pc.current) { 
              await pc.current.setRemoteDescription(new RTCSessionDescription(msg.sdp)); 
              makingOffer.current = false; 
            }
            break;

          case "ice":
            if (pc.current?.remoteDescription) {
              try { await pc.current.addIceCandidate(new RTCIceCandidate(msg.candidate)); } catch {}
            }
            break;

          case "peer_left":
            setPeerStatus("waiting"); 
            setStatusMsg(localIsHost ? "Guest left." : "Host left.");
            if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
            addMsg("Peer left the room.", "system");
            makingOffer.current = false;
            break;

          case "ping": sendSig({ type: "pong" }); break;
          default: break;
        }
      };

      ws.onerror = () => { 
        setPeerStatus("error"); 
        setStatusMsg("Signaling connection failed"); 
      };
    }

    init();
    return () => { 
      dead.current = true; 
      sig.current?.close(); 
      pc.current?.close(); 
      localStream.current?.getTracks().forEach(t => t.stop()); 
      recognitionRef.current?.stop(); 
    };
  }, [roomCode, isHost, addMsg, createPC, startOffer, sendSig, localIsHost]);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // --- Feature Logic ---
  const getSuggestion = useCallback(async (signWord) => {
    setIsThinking(true);
    try {
      const all  = messagesRef.current.filter(m => m.sender !== "system" && m.type !== "sign");
      const last = all.length > 0 ? [all[all.length - 1]] : [];
      const hist = (last.length > 0 && last[0].sender === "remote" && lastRepliedRef.current === last[0].id) ? [] : last;
      const res  = await fetch(`${API}/suggest`, { 
        method: "POST", 
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          sign_word: signWord, 
          history: hist.map(m => ({ sender: m.sender === "local" ? "me" : "them", text: m.text })) 
        }) 
      });
      const data = await res.json();
      setInputText(data.sentence?.trim() || signWord.charAt(0).toUpperCase() + signWord.slice(1));
      inputSourceRef.current = "sign";
    } catch { 
      setInputText(signWord.charAt(0).toUpperCase() + signWord.slice(1)); 
    } finally { 
      setIsThinking(false); 
    }
  }, []);

  const onWordDetected = useCallback(w => getSuggestion(w), [getSuggestion]);
  const getContext     = useCallback(() => messagesRef.current.filter(m => m.sender !== "system" && m.type !== "sign").slice(-3).map(m => m.text).join(" "), []);
  const { canvasRef: handCanvasRef, handsDetected, wsConnected, debugInfo } = useHandDetection(localVideoRef, camOn, onWordDetected, getContext);

  function toggleMic() {
    if (!localStream.current) return; 
    const next = !micOn;
    if (next && sttOn) { setSttOn(false); recognitionRef.current?.stop(); }
    localStream.current.getAudioTracks().forEach(t => t.enabled = next); 
    setMicOn(next);
  }

  function toggleCam() { 
    if (!localStream.current) return; 
    const next = !camOn; 
    localStream.current.getVideoTracks().forEach(t => t.enabled = next); 
    setCamOn(next); 
  }

  function toggleStt() {
    const next = !sttOn; setSttOn(next);
    if (next) {
      if (micOn) { setMicOn(false); localStream.current?.getAudioTracks().forEach(t => t.enabled = false); }
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) { addMsg("STT not supported.", "system"); setSttOn(false); return; }
      const r = new SR(); r.continuous = true; r.lang = "en-US";
      r.onresult = e => { setInputText(p => p+(p?" ":"")+e.results[e.results.length-1][0].transcript); };
      r.onend = () => { if (recognitionRef.current) try { r.start(); } catch {} };
      recognitionRef.current = r; r.start();
    } else { recognitionRef.current?.stop(); recognitionRef.current = null; }
  }

  function sendMessage() {
    const text = inputText.trim(); if (!text) return;
    const msgType = inputSourceRef.current === "sign" ? "sign" : "text";
    addMsg(text, "local", msgType);
    setInputText(""); inputSourceRef.current = "manual";
  }

  function handleLeave() {
    dead.current = true; onLeave();
  }

  const dotColor = { connecting:"#f0c040", waiting:"#f0c040", connected:"#4ecda4", error:"#e85d60" }[peerStatus] ?? "#f0c040";
  const remoteConnected = peerStatus === "connected";

  // --- Render ---
  return (
    <div className={styles.root}>
      <StarCanvas/>
      <svg className={styles.city} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 220" preserveAspectRatio="xMidYMax slice">
         <path fill="#0d0a28" d="M0,220 L0,180 L50,180 L50,160 L70,160 L70,155 L90,155 L90,160 L110,160 L110,175 L130,175 L130,155 L150,155 L150,140 L170,140 L170,155 L190,155 L190,175 L210,175 L210,160 L230,160 L230,145 L250,145 L250,160 L270,160 L270,175 L290,175 L290,155 L310,155 L310,140 L330,140 L330,120 L350,120 L350,140 L370,140 L370,155 L390,155 L390,170 L420,170 L420,155 L440,155 L440,140 L460,140 L460,120 L480,120 L480,140 L500,140 L500,155 L530,155 L530,170 L560,170 L560,155 L580,155 L580,140 L600,140 L600,120 L620,120 L620,100 L640,100 L640,120 L660,120 L660,140 L680,140 L680,155 L700,155 L700,170 L730,170 L730,155 L750,155 L750,135 L770,135 L770,120 L790,120 L790,135 L810,135 L810,155 L840,155 L840,170 L870,170 L870,155 L890,155 L890,140 L910,140 L910,120 L930,120 L930,140 L950,140 L950,155 L980,155 L980,170 L1010,170 L1010,155 L1030,155 L1030,140 L1050,140 L1050,120 L1070,120 L1070,100 L1090,100 L1090,120 L1110,120 L1110,140 L1130,140 L1130,160 L1160,160 L1160,175 L1190,175 L1190,160 L1210,160 L1210,145 L1230,145 L1230,160 L1250,160 L1250,175 L1280,175 L1280,160 L1300,160 L1300,145 L1320,145 L1320,160 L1340,160 L1340,175 L1370,175 L1370,180 L1440,180 L1440,220 Z"/>
      </svg>

      <header className={styles.topBar}>
        <div className={styles.topLeft}>
          <div className={styles.roomChip}>
            <span className={styles.roomChipIcon}><IconCam/></span>
            <span className={styles.roomCode}>{roomCode}</span>
            <span className={styles.statusDot} style={{background: dotColor}}/>
            <span className={styles.statusText} style={{color: dotColor}}>{statusMsg}</span>
          </div>
        </div>
        <div className={styles.topRight}>
          <button className={`${styles.ctrlBtn} ${sttOn?styles.ctrlOn:styles.ctrlOff}`} onClick={toggleStt}>{sttOn?<IconStt/>:<IconSttOff/>}</button>
          <button className={`${styles.ctrlBtn} ${micOn?styles.ctrlOn:styles.ctrlOff}`} onClick={toggleMic}>{micOn?<IconMic/>:<IconMicOff/>}</button>
          <button className={`${styles.ctrlBtn} ${camOn?styles.ctrlOn:styles.ctrlOff}`} onClick={toggleCam}>{camOn?<IconCam/>:<IconCamOff/>}</button>
          <button className={`${styles.ctrlBtn} ${styles.ctrlEnd}`} onClick={handleLeave}><IconPhone/></button>
        </div>
      </header>

      <div className={styles.main}>
        <div className={styles.videoRow}>
          <div className={styles.videoCard}>
            <video ref={localVideoRef} autoPlay playsInline muted className={`${styles.videoEl} ${camOn?styles.videoVisible:styles.videoHidden}`}/>
            {camOn && <canvas ref={handCanvasRef} className={styles.handCanvas}/>}
            {!camOn && <div className={styles.videoPlaceholder}><div className={styles.avatarCircle}><IconUser/></div></div>}
            <div className={styles.nameTag}>
              <span className={styles.nameTagMic}><IconMicSm/></span>
              {localIsHost ? "You (Host)" : "You (Guest)"}
            </div>
          </div>

          <div className={styles.videoCard}>
            <video ref={remoteVideoRef} autoPlay playsInline className={`${styles.videoEl} ${remoteConnected?styles.videoVisible:styles.videoHidden}`}/>
            {!remoteConnected && <div className={styles.videoPlaceholder}><div className={styles.avatarCircle}><IconUser/></div><p className={styles.waitingText}>{statusMsg}</p></div>}
            {remoteConnected && (
              <div className={styles.nameTag}>
                <span className={styles.nameTagMic}><IconMicSm/></span>
                {localIsHost ? "Guest" : "Host"}
              </div>
            )}
          </div>
        </div>

        <div className={styles.bottomSection}>
          <div className={styles.chatColumn}>
            <div className={styles.chatMessages}>
              {messages.map(m=>(
                <div key={m.id} className={m.sender==="system"?styles.bubbleSystem:m.sender==="local"?`${styles.bubble} ${styles.bubbleLocal}`:`${styles.bubble} ${styles.bubbleRemote}`}>
                  {m.text}
                  {m.sender!=="system"&&<span className={styles.bubbleTime}>{m.time}</span>}
                </div>
              ))}
              <div ref={chatEndRef}/>
            </div>
            <div className={styles.inputBar}>
              <input className={styles.inputField} placeholder="Type or sign a message…"
                value={inputText} onChange={e=>{setInputText(e.target.value);inputSourceRef.current="manual";}}
                onKeyDown={e=>{if(e.key==="Enter")sendMessage();}}/>
              <button className={styles.sendBtn} onClick={sendMessage}><IconSend/></button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}