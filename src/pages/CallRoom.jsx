import { useState, useRef, useEffect, useCallback } from "react";
import StarCanvas from "../components/StarCanvas";
import { useHandDetection } from "../hooks/useHandDetection";
import styles from "./CallRoom.module.css";

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
  const [micOn,      setMicOn]      = useState(false);
  const [camOn,      setCamOn]      = useState(false);
  const [sttOn,      setSttOn]      = useState(false);
  const [peerStatus, setPeerStatus] = useState("connecting");
  const [statusMsg,  setStatusMsg]  = useState("Connecting to server…");
  const [messages,   setMessages]   = useState([]);
  const [inputText,  setInputText]  = useState("");
  const [isThinking, setIsThinking] = useState(false);

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
      if (remoteVideoRef.current) { remoteVideoRef.current.srcObject = e.streams[0]; remoteVideoRef.current.play().catch(() => {}); }
      setPeerStatus("connected"); setStatusMsg("Live");
      addMsg("Connected! You are now in the same room.", "system");
    };
    conn.onicecandidate = (e) => { if (e.candidate) sendSig({ type: "ice", candidate: e.candidate }); };
    conn.onconnectionstatechange = () => {
      if (dead.current) return;
      const s = conn.connectionState;
      if (s === "failed") { setPeerStatus("error"); setStatusMsg("Connection failed — refresh both devices"); addMsg("WebRTC failed. Please refresh.", "system"); }
      if (s === "disconnected") { setPeerStatus("waiting"); setStatusMsg("Other person disconnected"); if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null; addMsg("The other person left.", "system"); }
    };
    pc.current = conn; return conn;
  }, [addMsg, sendSig]);

  const startOffer = useCallback(async () => {
    if (makingOffer.current) return;
    makingOffer.current = true;
    setStatusMsg("Found host — establishing connection…");
    try {
      const conn = createPC();
      const offer = await conn.createOffer();
      await conn.setLocalDescription(offer);
      sendSig({ type: "offer", sdp: conn.localDescription });
    } catch (e) { console.error("offer failed:", e); makingOffer.current = false; }
  }, [createPC, sendSig]);

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
        setPeerStatus("error"); setStatusMsg("Camera/mic denied — allow permissions and refresh"); return;
      }

      const safeRoom = roomCode.replace(/[^a-zA-Z0-9-]/g, "");
      const role     = isHost ? "host" : "guest";
      const url      = `${WS_API}/ws/signal/${safeRoom}/${role}`;
      console.log(`[signal] Connecting as ${role} to room: ${safeRoom}`);

      const ws = new WebSocket(url);
      sig.current = ws;

      ws.onopen = () => {
        if (isHost) { setPeerStatus("waiting"); setStatusMsg(`Waiting for guest — share code "${roomCode}"`); addMsg(`Room ready as HOST. Share code "${roomCode}" with the guest.`, "system"); }
        else        { setPeerStatus("connecting"); setStatusMsg("Connected — looking for host…"); addMsg("Joined as GUEST. Looking for host…", "system"); }
      };

      ws.onmessage = async (e) => {
        if (dead.current) return;
        let msg; try { msg = JSON.parse(e.data); } catch { return; }
        console.log("[signal] ←", msg.type);
        switch (msg.type) {
          case "ready":
            if (!isHost) await startOffer();
            break;
          case "wait_for_host":
            setStatusMsg("Host not in room yet — waiting…");
            addMsg("Host hasn't joined yet. They need to open the room first.", "system");
            break;
          case "guest_joined":
            if (isHost) { setPeerStatus("connecting"); setStatusMsg("Guest found — connecting…"); addMsg("Someone is joining…", "system"); }
            break;
          case "offer":
            if (isHost) {
              const conn = createPC();
              await conn.setRemoteDescription(new RTCSessionDescription(msg.sdp));
              const answer = await conn.createAnswer();
              await conn.setLocalDescription(answer);
              sendSig({ type: "answer", sdp: conn.localDescription });
            }
            break;
          case "answer":
            if (!isHost && pc.current) { await pc.current.setRemoteDescription(new RTCSessionDescription(msg.sdp)); makingOffer.current = false; }
            break;
          case "ice":
            if (pc.current?.remoteDescription) try { await pc.current.addIceCandidate(new RTCIceCandidate(msg.candidate)); } catch {}
            break;
          case "peer_left":
            setPeerStatus("waiting"); setStatusMsg(isHost ? "Guest left." : "Host left.");
            if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
            addMsg(isHost ? "Guest left the room." : "Host left the room.", "system");
            makingOffer.current = false;
            break;
          case "ping": sendSig({ type: "pong" }); break;
          default: break;
        }
      };

      ws.onerror = () => { if (dead.current) return; setPeerStatus("error"); setStatusMsg("Cannot reach server — check VITE_API_URL in Vercel"); addMsg("Cannot connect to signaling server.", "system"); };
      ws.onclose = (e) => { if (dead.current || e.code === 1000) return; setStatusMsg("Server disconnected — refresh to reconnect"); };
    }
    init();
    return () => { dead.current = true; sig.current?.close(); pc.current?.close(); localStream.current?.getTracks().forEach(t => t.stop()); recognitionRef.current?.stop(); };
  }, []); // eslint-disable-line

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const getSuggestion = useCallback(async (signWord) => {
    setIsThinking(true);
    try {
      const all  = messagesRef.current.filter(m => m.sender !== "system" && m.type !== "sign");
      const last = all.length > 0 ? [all[all.length - 1]] : [];
      const hist = (last.length > 0 && last[0].sender === "remote" && lastRepliedRef.current === last[0].id) ? [] : last;
      const res  = await fetch(`${API}/suggest`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sign_word: signWord, history: hist.map(m => ({ sender: m.sender === "local" ? "me" : "them", text: m.text })) }) });
      const data = await res.json();
      setInputText(data.sentence?.trim() || signWord.charAt(0).toUpperCase() + signWord.slice(1));
      inputSourceRef.current = "sign";
    } catch { setInputText(signWord.charAt(0).toUpperCase() + signWord.slice(1)); inputSourceRef.current = "sign"; }
    finally { setIsThinking(false); }
  }, []);

  const onWordDetected = useCallback(w => getSuggestion(w), [getSuggestion]);
  const getContext     = useCallback(() => messagesRef.current.filter(m => m.sender !== "system" && m.type !== "sign").slice(-3).map(m => m.text).join(" "), []);
  const { canvasRef: handCanvasRef, handsDetected, wsConnected, debugInfo } = useHandDetection(localVideoRef, camOn, onWordDetected, getContext);

  function toggleMic() {
    if (!localStream.current) return; const next = !micOn;
    if (next && sttOn) { setSttOn(false); recognitionRef.current?.stop(); recognitionRef.current = null; }
    localStream.current.getAudioTracks().forEach(t => t.enabled = next); setMicOn(next);
  }
  function toggleCam() { if (!localStream.current) return; const next = !camOn; localStream.current.getVideoTracks().forEach(t => t.enabled = next); setCamOn(next); }
  function toggleStt() {
    const next = !sttOn; setSttOn(next);
    if (next) {
      if (micOn) { setMicOn(false); localStream.current?.getAudioTracks().forEach(t => t.enabled = false); }
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) { addMsg("Speech recognition not supported.", "system"); setSttOn(false); return; }
      const r = new SR(); r.continuous = true; r.interimResults = false; r.lang = "en-US";
      r.onresult = e => { const t = e.results[e.results.length-1][0].transcript; setInputText(p => p+(p?" ":"")+t); };
      r.onerror = () => setSttOn(false); r.onend = () => { if (recognitionRef.current) try { r.start(); } catch {} };
      recognitionRef.current = r; try { r.start(); } catch { addMsg("Failed to start STT.", "system"); setSttOn(false); recognitionRef.current = null; }
    } else { recognitionRef.current?.stop(); recognitionRef.current = null; }
  }
  function sendMessage() {
    const text = inputText.trim(); if (!text) return;
    const msgType = inputSourceRef.current === "sign" ? "sign" : "text";
    addMsg(text, "local", msgType);
    if (msgType === "sign") { const all = messagesRef.current.filter(m => m.sender !== "system" && m.type !== "sign"); if (all.length > 0 && all[all.length-1].sender === "remote") lastRepliedRef.current = all[all.length-1].id; }
    setInputText(""); inputSourceRef.current = "manual";
  }
  function handleLeave() {
    dead.current = true; sig.current?.close(); pc.current?.close();
    localStream.current?.getTracks().forEach(t => t.stop()); recognitionRef.current?.stop(); setSttOn(false); onLeave();
  }

  const dotColor = { connecting:"#f0c040", waiting:"#f0c040", connected:"#4ecda4", error:"#e85d60" }[peerStatus] ?? "#f0c040";
  const remoteConnected = peerStatus === "connected";

  return (
    <div className={styles.root}>
      <StarCanvas/>
      <svg className={styles.city} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 220" preserveAspectRatio="xMidYMax slice">
        <defs><linearGradient id="cg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#1a1050" stopOpacity="0.7"/><stop offset="100%" stopColor="#0a0820" stopOpacity="1"/></linearGradient></defs>
        <path fill="url(#cg)" opacity="0.5" d="M0,220 L0,160 L40,160 L40,140 L60,140 L60,120 L80,120 L80,100 L100,100 L100,80 L120,80 L120,100 L140,100 L140,60 L160,60 L160,100 L180,100 L180,80 L200,80 L200,140 L240,140 L240,120 L260,120 L260,100 L280,100 L280,80 L300,80 L300,60 L320,60 L320,80 L340,80 L340,100 L360,100 L360,120 L380,120 L380,80 L400,80 L400,60 L420,60 L420,40 L440,40 L440,60 L460,60 L460,80 L480,80 L480,100 L520,100 L520,80 L540,80 L540,60 L560,60 L560,80 L580,80 L580,100 L600,100 L600,80 L620,80 L620,60 L640,60 L640,80 L680,80 L680,60 L700,60 L700,40 L720,40 L720,60 L740,60 L740,80 L760,80 L760,100 L800,100 L800,80 L820,80 L820,60 L840,60 L840,40 L860,40 L860,60 L880,60 L880,80 L900,80 L900,100 L940,100 L940,120 L960,120 L960,100 L980,100 L980,80 L1000,80 L1000,60 L1020,60 L1020,80 L1060,80 L1060,100 L1080,100 L1080,80 L1100,80 L1100,60 L1120,60 L1120,40 L1140,40 L1140,60 L1160,60 L1160,80 L1200,80 L1200,100 L1220,100 L1220,80 L1240,80 L1240,60 L1260,60 L1260,80 L1280,80 L1280,100 L1300,100 L1300,80 L1320,80 L1320,60 L1340,60 L1340,80 L1360,80 L1360,100 L1380,100 L1380,140 L1440,140 L1440,220 Z"/>
        <path fill="#0d0a28" d="M0,220 L0,180 L50,180 L50,160 L70,160 L70,155 L90,155 L90,160 L110,160 L110,175 L130,175 L130,155 L150,155 L150,140 L170,140 L170,155 L190,155 L190,175 L210,175 L210,160 L230,160 L230,145 L250,145 L250,160 L270,160 L270,175 L290,175 L290,155 L310,155 L310,140 L330,140 L330,120 L350,120 L350,140 L370,140 L370,155 L390,155 L390,170 L420,170 L420,155 L440,155 L440,140 L460,140 L460,120 L480,120 L480,140 L500,140 L500,155 L530,155 L530,170 L560,170 L560,155 L580,155 L580,140 L600,140 L600,120 L620,120 L620,100 L640,100 L640,120 L660,120 L660,140 L680,140 L680,155 L700,155 L700,170 L730,170 L730,155 L750,155 L750,135 L770,135 L770,120 L790,120 L790,135 L810,135 L810,155 L840,155 L840,170 L870,170 L870,155 L890,155 L890,140 L910,140 L910,120 L930,120 L930,140 L950,140 L950,155 L980,155 L980,170 L1010,170 L1010,155 L1030,155 L1030,140 L1050,140 L1050,120 L1070,120 L1070,100 L1090,100 L1090,120 L1110,120 L1110,140 L1130,140 L1130,160 L1160,160 L1160,175 L1190,175 L1190,160 L1210,160 L1210,145 L1230,145 L1230,160 L1250,160 L1250,175 L1280,175 L1280,160 L1300,160 L1300,145 L1320,145 L1320,160 L1340,160 L1340,175 L1370,175 L1370,180 L1440,180 L1440,220 Z"/>
      </svg>

      <header className={styles.topBar}>
        <div className={styles.topLeft}>
          <div className={styles.roomChip}>
            <span className={styles.roomChipIcon}><IconCam/></span>
            <span className={styles.roomCode}>{roomCode}</span>
            <span className={styles.statusDot} style={{background: dotColor}}/>
            <span className={styles.statusText} style={{color: dotColor}}>{statusMsg}</span>
            <button className={styles.chipBtn}><IconDots/></button>
          </div>
        </div>
        <div className={styles.topRight}>
          <button className={`${styles.ctrlBtn} ${sttOn?styles.ctrlOn:styles.ctrlOff}`} onClick={toggleStt}>{sttOn?<IconStt/>:<IconSttOff/>}</button>
          <button className={`${styles.ctrlBtn} ${micOn?styles.ctrlOn:styles.ctrlOff}`} onClick={toggleMic}>{micOn?<IconMic/>:<IconMicOff/>}</button>
          <button className={`${styles.ctrlBtn} ${camOn?styles.ctrlOn:styles.ctrlOff}`} onClick={toggleCam}>{camOn?<IconCam/>:<IconCamOff/>}</button>
          <button className={`${styles.ctrlBtn} ${styles.ctrlEnd}`} onClick={handleLeave}><IconPhone/></button>
          <button className={`${styles.ctrlBtn} ${styles.ctrlOff}`}><IconMenu/></button>
        </div>
      </header>

      <div className={styles.main}>
        <div className={styles.videoRow}>
          <div className={styles.videoCard}>
            <video ref={localVideoRef} autoPlay playsInline muted className={`${styles.videoEl} ${camOn?styles.videoVisible:styles.videoHidden}`}/>
            {camOn && <canvas ref={handCanvasRef} className={styles.handCanvas}/>}
            {camOn && debugInfo && (
              <div className={styles.debugPanel}>
                <div className={styles.debugRow}><span className={styles.debugFinger}>status</span><span className={styles.debugVal} style={{color:debugInfo.status==="accepted"?"#4ecda4":"rgba(200,190,255,0.6)"}}>{debugInfo.status}</span></div>
                {debugInfo.top_word && <div className={styles.debugRow}><span className={styles.debugFinger}>sees</span><span className={styles.debugVal}>{debugInfo.top_word} {Math.round((debugInfo.confidence||0)*100)}%</span></div>}
                {debugInfo.buffer !== undefined && <div className={styles.debugRow}><span className={styles.debugFinger}>buf</span><span className={styles.debugVal}>{debugInfo.buffer}/{debugInfo.needed}</span></div>}
                <div className={styles.debugSign}>{debugInfo?.status==="accepted"?`✓ ${debugInfo.word}`:handsDetected?"detecting…":"show hand"}</div>
              </div>
            )}
            {!camOn && <div className={styles.videoPlaceholder}><div className={styles.avatarCircle}><IconUser/></div></div>}
            <div className={styles.nameTag}><span className={styles.nameTagMic}><IconMicSm/></span>{isHost?"You (Host)":"You (Guest)"}</div>
            {camOn && <div className={styles.liveTag} style={{background:wsConnected?"#4ecda4":"#e85d60"}}>{wsConnected?"LIVE":"NO SERVER"}</div>}
          </div>

          <div className={styles.videoCard}>
            <video ref={remoteVideoRef} autoPlay playsInline className={`${styles.videoEl} ${remoteConnected?styles.videoVisible:styles.videoHidden}`}/>
            {!remoteConnected && <div className={styles.videoPlaceholder}><div className={styles.avatarCircle}><IconUser/></div><p className={styles.waitingText}>{statusMsg}</p></div>}
            {remoteConnected && <div className={styles.nameTag}><span className={styles.nameTagMic}><IconMicSm/></span>{isHost?"Guest":"Host"}</div>}
          </div>
        </div>

        <div className={styles.bottomSection}>
          <div className={styles.avatarSpace}><div className={styles.avatarSpaceInner}><span className={styles.avatarSpaceLabel}>Avatar space</span></div></div>
          <div className={styles.chatColumn}>
            {peerStatus !== "connected" && (
              <div className={styles.testBar}>
                <span className={styles.testLabel}>🧪 Test signs</span>
                {["Hi! How are you?","Are you hungry?","Where are you going?","Feeling okay?"].map(msg=>(
                  <button key={msg} className={styles.testBtn} onClick={()=>addMsg(msg,"remote")}>{msg}</button>
                ))}
              </div>
            )}
            <div className={styles.chatMessages}>
              {messages.map(m=>(
                <div key={m.id} className={m.sender==="system"?styles.bubbleSystem:m.sender==="local"?`${styles.bubble} ${styles.bubbleLocal}`:`${styles.bubble} ${styles.bubbleRemote}`}>
                  {m.type==="sign"&&<span style={{marginRight:6,opacity:0.8}}>🤚</span>}
                  {m.text}
                  {m.sender!=="system"&&<span className={styles.bubbleTime}>{m.time}</span>}
                </div>
              ))}
              <div ref={chatEndRef}/>
            </div>
            <div className={styles.inputBar}>
              <button className={styles.inputIconBtn}><IconPlus/></button>
              <button className={`${styles.inputIconBtn} ${sttOn?styles.inputIconActive:""}`} onClick={toggleStt}>{sttOn?<IconMic/>:<IconMicOff/>}</button>
              <div className={styles.inputWrapper}>
                {isThinking&&<div className={styles.thinkingBadge}><span/><span/><span/></div>}
                <input className={styles.inputField} placeholder={isThinking?"Generating sentence…":"Type or sign a message…"}
                  value={inputText} onChange={e=>{setInputText(e.target.value);inputSourceRef.current="manual";}}
                  onKeyDown={e=>{if(e.key==="Enter")sendMessage();}}/>
              </div>
              <button className={styles.inputIconBtn}><IconEmoji/></button>
              <button className={styles.sendBtn} onClick={sendMessage}><IconSend/></button>
            </div>
          </div>
          <div className={styles.avatarSpace}><div className={styles.avatarSpaceInner}><span className={styles.avatarSpaceLabel}>Avatar space</span></div></div>
        </div>
      </div>
    </div>
  );
}