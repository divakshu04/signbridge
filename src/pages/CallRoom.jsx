import { useState, useRef, useEffect, useCallback } from "react";
import StarCanvas from "../components/StarCanvas";
import { useHandDetection } from "../hooks/useHandDetection";
import styles from "./CallRoom.module.css";

/* ── Icons ── */
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

/* ── ICE servers: STUN + free TURN (Open Relay) ── */
const ICE_SERVERS = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
    { urls: "turn:openrelay.metered.ca:80",       username: "openrelayproject", credential: "openrelayproject" },
    { urls: "turn:openrelay.metered.ca:443",      username: "openrelayproject", credential: "openrelayproject" },
    { urls: "turn:openrelay.metered.ca:443?transport=tcp", username: "openrelayproject", credential: "openrelayproject" },
  ],
};

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000")
  .replace(/\/$/, "");

const WS_BASE = API.replace(/^http/, "ws");

export default function CallRoom({ roomCode, isHost, onLeave }) {
  const [micOn,      setMicOn]      = useState(false);
  const [camOn,      setCamOn]      = useState(false);
  const [sttOn,      setSttOn]      = useState(false);
  const [peerStatus, setPeerStatus] = useState("connecting");
  const [messages,   setMessages]   = useState([]);
  const [inputText,  setInputText]  = useState("");
  const [remoteUser, setRemoteUser] = useState(null);
  const [isThinking, setIsThinking] = useState(false);

  const localVideoRef  = useRef(null);
  const remoteVideoRef = useRef(null);
  const localStream    = useRef(null);
  const pcRef          = useRef(null);   // RTCPeerConnection
  const sigRef         = useRef(null);   // signaling WebSocket
  const chatEndRef     = useRef(null);
  const messagesRef    = useRef([]);
  const inputSourceRef = useRef("manual");
  const lastRepliedRef = useRef(null);
  const recognitionRef = useRef(null);
  const deadRef        = useRef(false);  // component unmounted

  /* ── Add message ── */
  const addMsg = useCallback((text, sender, type = "text") => {
    const msg = { id: Date.now() + Math.random(), text, sender, type,
                  time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) };
    setMessages(prev => { const u = [...prev, msg]; messagesRef.current = u; return u; });
  }, []);

  /* ── Send a signal through the signaling WebSocket ── */
  function sendSignal(obj) {
    const ws = sigRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }

  /* ── Create RTCPeerConnection ── */
  function createPC() {
    const pc = new RTCPeerConnection(ICE_SERVERS);

    // Send our local tracks to the remote peer
    if (localStream.current) {
      localStream.current.getTracks().forEach(t => pc.addTrack(t, localStream.current));
    }

    // When we get remote tracks, show them
    pc.ontrack = (e) => {
      if (deadRef.current) return;
      const [stream] = e.streams;
      if (remoteVideoRef.current) {
        remoteVideoRef.current.srcObject = stream;
        remoteVideoRef.current.play().catch(() => {});
      }
      setPeerStatus("connected");
      setRemoteUser(isHost ? "Guest" : "Host");
      addMsg("Connected!", "system");
    };

    // Send ICE candidates to the other peer via signaling
    pc.onicecandidate = (e) => {
      if (e.candidate) sendSignal({ type: "ice", candidate: e.candidate });
    };

    pc.onconnectionstatechange = () => {
      if (deadRef.current) return;
      const s = pc.connectionState;
      console.log("WebRTC state:", s);
      if (s === "disconnected" || s === "failed" || s === "closed") {
        setPeerStatus("waiting");
        setRemoteUser(null);
        if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
        if (s !== "closed") addMsg("Call ended.", "system");
      }
    };

    pcRef.current = pc;
    return pc;
  }

  /* ── Main setup ── */
  useEffect(() => {
    deadRef.current = false;

    async function init() {
      /* 1. Get local media */
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        stream.getVideoTracks().forEach(t => t.enabled = false);
        stream.getAudioTracks().forEach(t  => t.enabled = false);
        localStream.current = stream;
        if (localVideoRef.current) localVideoRef.current.srcObject = stream;
      } catch {
        setPeerStatus("error");
        addMsg("Camera/mic access denied — allow permissions and refresh.", "system");
        return;
      }

      /* 2. Connect to YOUR signaling server on Render */
      const sigUrl = `${WS_BASE}/ws/signal/${encodeURIComponent(roomCode)}/${isHost ? "host" : "guest"}`;
      console.log("Connecting to signaling:", sigUrl);

      const sig = new WebSocket(sigUrl);
      sigRef.current = sig;

      sig.onopen = () => {
        console.log("✓ Signaling connected");
        if (isHost) {
          setPeerStatus("waiting");
          addMsg(`Room ready — share code "${roomCode}" with the other person.`, "system");
        } else {
          addMsg("Waiting for host…", "system");
        }
      };

      sig.onmessage = async (e) => {
        if (deadRef.current) return;
        let msg;
        try { msg = JSON.parse(e.data); } catch { return; }

        /* "ready" = both peers are in the room, guest starts the offer */
        if (msg.type === "ready") {
          if (!isHost) {
            addMsg("Host found — connecting…", "system");
            const pc = createPC();
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            sendSignal({ type: "offer", sdp: pc.localDescription });
          }
          return;
        }

        if (msg.type === "offer" && isHost) {
          const pc = createPC();
          await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp));
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          sendSignal({ type: "answer", sdp: pc.localDescription });
          setPeerStatus("connecting");
          return;
        }

        if (msg.type === "answer" && !isHost) {
          await pcRef.current?.setRemoteDescription(new RTCSessionDescription(msg.sdp));
          return;
        }

        if (msg.type === "ice") {
          try {
            await pcRef.current?.addIceCandidate(new RTCIceCandidate(msg.candidate));
          } catch {}
          return;
        }
      };

      sig.onerror = () => {
        if (deadRef.current) return;
        addMsg("Could not reach signaling server — check VITE_API_URL.", "system");
        setPeerStatus("error");
      };

      sig.onclose = () => {
        if (deadRef.current) return;
        console.warn("Signaling WS closed");
      };
    }

    init();

    return () => {
      deadRef.current = true;
      sigRef.current?.close();
      pcRef.current?.close();
      localStream.current?.getTracks().forEach(t => t.stop());
      recognitionRef.current?.stop();
    };
  }, []); // eslint-disable-line

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  /* ── Groq suggestion ── */
  const getSuggestion = useCallback(async (signWord) => {
    setIsThinking(true);
    try {
      const allMsgs = messagesRef.current.filter(m => m.sender !== "system" && m.type !== "sign");
      const history = allMsgs.length > 0 ? [allMsgs[allMsgs.length - 1]] : [];
      let historyForGroq = history;
      if (history.length > 0 && history[0].sender === "remote" && lastRepliedRef.current === history[0].id)
        historyForGroq = [];

      const res = await fetch(`${API}/suggest`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sign_word: signWord,
          history: historyForGroq.map(m => ({ sender: m.sender === "local" ? "me" : "them", text: m.text })) }),
      });
      const data = await res.json();
      setInputText(data.sentence?.trim() || signWord.charAt(0).toUpperCase() + signWord.slice(1));
      inputSourceRef.current = "sign";
    } catch {
      setInputText(signWord.charAt(0).toUpperCase() + signWord.slice(1));
      inputSourceRef.current = "sign";
    } finally { setIsThinking(false); }
  }, []);

  const onWordDetected  = useCallback((word) => getSuggestion(word), [getSuggestion]);
  const getContext      = useCallback(() =>
    messagesRef.current.filter(m => m.sender !== "system" && m.type !== "sign").slice(-3).map(m => m.text).join(" ")
  , []);

  const { canvasRef: handCanvasRef, handsDetected, wsConnected, debugInfo } =
    useHandDetection(localVideoRef, camOn, onWordDetected, getContext);

  /* ── Controls ── */
  function toggleMic() {
    if (!localStream.current) return;
    const next = !micOn;
    if (next && sttOn) { setSttOn(false); recognitionRef.current?.stop(); recognitionRef.current = null; }
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
    const next = !sttOn;
    setSttOn(next);
    if (next) {
      if (micOn) { setMicOn(false); localStream.current?.getAudioTracks().forEach(t => t.enabled = false); }
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) { addMsg("Speech recognition not supported.", "system"); setSttOn(false); return; }
      const r = new SR();
      r.continuous = true; r.interimResults = false; r.lang = "en-US";
      r.onresult = (e) => { const t = e.results[e.results.length-1][0].transcript; setInputText(p => p + (p?" ":"") + t); };
      r.onerror  = () => setSttOn(false);
      r.onend    = () => { if (recognitionRef.current) try { r.start(); } catch {} };
      recognitionRef.current = r;
      try { r.start(); } catch { addMsg("Failed to start speech recognition.", "system"); setSttOn(false); recognitionRef.current = null; }
    } else { recognitionRef.current?.stop(); recognitionRef.current = null; }
  }

  function sendMessage() {
    const text = inputText.trim(); if (!text) return;
    const msgType = inputSourceRef.current === "sign" ? "sign" : "text";
    addMsg(text, "local", msgType);
    if (msgType === "sign") {
      const all = messagesRef.current.filter(m => m.sender !== "system" && m.type !== "sign");
      if (all.length > 0 && all[all.length-1].sender === "remote") lastRepliedRef.current = all[all.length-1].id;
    }
    setInputText(""); inputSourceRef.current = "manual";
  }

  function handleLeave() {
    deadRef.current = true;
    sigRef.current?.close();
    pcRef.current?.close();
    localStream.current?.getTracks().forEach(t => t.stop());
    recognitionRef.current?.stop();
    setSttOn(false); onLeave();
  }

  const statusInfo = {
    connecting: { color: "#f0c040", text: "Connecting…"       },
    waiting:    { color: "#4ecda4", text: "Waiting for guest…" },
    connected:  { color: "#4ecda4", text: "Live"              },
    error:      { color: "#e85d60", text: "Error"             },
  }[peerStatus] ?? { color: "#f0c040", text: "…" };

  const remoteConnected = peerStatus === "connected";

  return (
    <div className={styles.root}>
      <StarCanvas />

      {/* City */}
      <svg className={styles.city} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 220" preserveAspectRatio="xMidYMax slice">
        <defs><linearGradient id="cg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#1a1050" stopOpacity="0.7"/><stop offset="100%" stopColor="#0a0820" stopOpacity="1"/></linearGradient></defs>
        <path fill="url(#cg)" opacity="0.5" d="M0,220 L0,160 L40,160 L40,140 L60,140 L60,120 L80,120 L80,100 L100,100 L100,80 L120,80 L120,100 L140,100 L140,60 L160,60 L160,100 L180,100 L180,80 L200,80 L200,140 L240,140 L240,120 L260,120 L260,100 L280,100 L280,80 L300,80 L300,60 L320,60 L320,80 L340,80 L340,100 L360,100 L360,120 L380,120 L380,80 L400,80 L400,60 L420,60 L420,40 L440,40 L440,60 L460,60 L460,80 L480,80 L480,100 L520,100 L520,80 L540,80 L540,60 L560,60 L560,80 L580,80 L580,100 L600,100 L600,80 L620,80 L620,60 L640,60 L640,80 L680,80 L680,60 L700,60 L700,40 L720,40 L720,60 L740,60 L740,80 L760,80 L760,100 L800,100 L800,80 L820,80 L820,60 L840,60 L840,40 L860,40 L860,60 L880,60 L880,80 L900,80 L900,100 L940,100 L940,120 L960,120 L960,100 L980,100 L980,80 L1000,80 L1000,60 L1020,60 L1020,80 L1060,80 L1060,100 L1080,100 L1080,80 L1100,80 L1100,60 L1120,60 L1120,40 L1140,40 L1140,60 L1160,60 L1160,80 L1200,80 L1200,100 L1220,100 L1220,80 L1240,80 L1240,60 L1260,60 L1260,80 L1280,80 L1280,100 L1300,100 L1300,80 L1320,80 L1320,60 L1340,60 L1340,80 L1360,80 L1360,100 L1380,100 L1380,140 L1440,140 L1440,220 Z"/>
        <path fill="#0d0a28" d="M0,220 L0,180 L50,180 L50,160 L70,160 L70,155 L90,155 L90,160 L110,160 L110,175 L130,175 L130,155 L150,155 L150,140 L170,140 L170,155 L190,155 L190,175 L210,175 L210,160 L230,160 L230,145 L250,145 L250,160 L270,160 L270,175 L290,175 L290,155 L310,155 L310,140 L330,140 L330,120 L350,120 L350,140 L370,140 L370,155 L390,155 L390,170 L420,170 L420,155 L440,155 L440,140 L460,140 L460,120 L480,120 L480,140 L500,140 L500,155 L530,155 L530,170 L560,170 L560,155 L580,155 L580,140 L600,140 L600,120 L620,120 L620,100 L640,100 L640,120 L660,120 L660,140 L680,140 L680,155 L700,155 L700,170 L730,170 L730,155 L750,155 L750,135 L770,135 L770,120 L790,120 L790,135 L810,135 L810,155 L840,155 L840,170 L870,170 L870,155 L890,155 L890,140 L910,140 L910,120 L930,120 L930,140 L950,140 L950,155 L980,155 L980,170 L1010,170 L1010,155 L1030,155 L1030,140 L1050,140 L1050,120 L1070,120 L1070,100 L1090,100 L1090,120 L1110,120 L1110,140 L1130,140 L1130,160 L1160,160 L1160,175 L1190,175 L1190,160 L1210,160 L1210,145 L1230,145 L1230,160 L1250,160 L1250,175 L1280,175 L1280,160 L1300,160 L1300,145 L1320,145 L1320,160 L1340,160 L1340,175 L1370,175 L1370,180 L1440,180 L1440,220 Z"/>
      </svg>

      {/* TOP BAR */}
      <header className={styles.topBar}>
        <div className={styles.topLeft}>
          <div className={styles.roomChip}>
            <span className={styles.roomChipIcon}><IconCam /></span>
            <span className={styles.roomCode}>{roomCode}</span>
            <span className={styles.statusDot} style={{ background: statusInfo.color }} />
            <span className={styles.statusText} style={{ color: statusInfo.color }}>{statusInfo.text}</span>
            <button className={styles.chipBtn}><IconDots /></button>
          </div>
        </div>
        <div className={styles.topRight}>
          <button className={`${styles.ctrlBtn} ${sttOn ? styles.ctrlOn : styles.ctrlOff}`} onClick={toggleStt}>
            {sttOn ? <IconStt /> : <IconSttOff />}
          </button>
          <button className={`${styles.ctrlBtn} ${micOn ? styles.ctrlOn : styles.ctrlOff}`} onClick={toggleMic}>
            {micOn ? <IconMic /> : <IconMicOff />}
          </button>
          <button className={`${styles.ctrlBtn} ${camOn ? styles.ctrlOn : styles.ctrlOff}`} onClick={toggleCam}>
            {camOn ? <IconCam /> : <IconCamOff />}
          </button>
          <button className={`${styles.ctrlBtn} ${styles.ctrlEnd}`} onClick={handleLeave}><IconPhone /></button>
          <button className={`${styles.ctrlBtn} ${styles.ctrlOff}`}><IconMenu /></button>
        </div>
      </header>

      {/* MAIN */}
      <div className={styles.main}>
        <div className={styles.videoRow}>

          {/* Local */}
          <div className={styles.videoCard}>
            <video ref={localVideoRef} autoPlay playsInline muted
              className={`${styles.videoEl} ${camOn ? styles.videoVisible : styles.videoHidden}`} />
            {camOn && <canvas ref={handCanvasRef} className={styles.handCanvas} />}
            {camOn && debugInfo && (
              <div className={styles.debugPanel}>
                <div className={styles.debugRow}>
                  <span className={styles.debugFinger}>status</span>
                  <span className={styles.debugVal} style={{ color: debugInfo.status === "accepted" ? "#4ecda4" : "rgba(200,190,255,0.6)" }}>{debugInfo.status}</span>
                </div>
                {debugInfo.top_word && (
                  <div className={styles.debugRow}>
                    <span className={styles.debugFinger}>sees</span>
                    <span className={styles.debugVal}>{debugInfo.top_word} {Math.round((debugInfo.confidence||0)*100)}%</span>
                  </div>
                )}
                {debugInfo.buffer !== undefined && (
                  <div className={styles.debugRow}>
                    <span className={styles.debugFinger}>buf</span>
                    <span className={styles.debugVal}>{debugInfo.buffer}/{debugInfo.needed}</span>
                  </div>
                )}
                <div className={styles.debugSign}>
                  {debugInfo?.status === "accepted" ? `✓ ${debugInfo.word}` : handsDetected ? "detecting…" : "show hand"}
                </div>
              </div>
            )}
            {!camOn && <div className={styles.videoPlaceholder}><div className={styles.avatarCircle}><IconUser /></div></div>}
            <div className={styles.nameTag}>
              <span className={styles.nameTagMic}><IconMicSm /></span>
              {isHost ? "You (Host)" : "You (Guest)"}
            </div>
            {camOn && (
              <div className={styles.liveTag} style={{ background: wsConnected ? "#4ecda4" : "#e85d60" }}>
                {wsConnected ? "LIVE" : "NO SERVER"}
              </div>
            )}
          </div>

          {/* Remote */}
          <div className={styles.videoCard}>
            <video ref={remoteVideoRef} autoPlay playsInline
              className={`${styles.videoEl} ${remoteConnected ? styles.videoVisible : styles.videoHidden}`} />
            {!remoteConnected && (
              <div className={styles.videoPlaceholder}>
                <div className={styles.avatarCircle}><IconUser /></div>
                <p className={styles.waitingText}>
                  {peerStatus === "error" ? "Connection failed" : peerStatus === "connecting" ? "Connecting…" : "Waiting to join…"}
                </p>
              </div>
            )}
            {remoteConnected && (
              <div className={styles.nameTag}>
                <span className={styles.nameTagMic}><IconMicSm /></span>
                {remoteUser || "Remote"}
              </div>
            )}
          </div>
        </div>

        {/* Bottom */}
        <div className={styles.bottomSection}>
          <div className={styles.avatarSpace}><div className={styles.avatarSpaceInner}><span className={styles.avatarSpaceLabel}>Avatar space</span></div></div>

          <div className={styles.chatColumn}>
            {peerStatus !== "connected" && (
              <div className={styles.testBar}>
                <span className={styles.testLabel}>🧪 Test</span>
                {["Hi! How are you?","Are you hungry?","Where are you going?","Feeling okay?"].map(msg => (
                  <button key={msg} className={styles.testBtn} onClick={() => addMsg(msg, "remote")}>{msg}</button>
                ))}
              </div>
            )}

            <div className={styles.chatMessages}>
              {messages.map(m => (
                <div key={m.id} className={
                  m.sender === "system" ? styles.bubbleSystem :
                  m.sender === "local"  ? `${styles.bubble} ${styles.bubbleLocal}` :
                                          `${styles.bubble} ${styles.bubbleRemote}`
                }>
                  {m.type === "sign" && <span style={{ marginRight: 6, opacity: 0.8 }}>🤚</span>}
                  {m.text}
                  {m.sender !== "system" && <span className={styles.bubbleTime}>{m.time}</span>}
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>

            <div className={styles.inputBar}>
              <button className={styles.inputIconBtn}><IconPlus /></button>
              <button className={`${styles.inputIconBtn} ${sttOn ? styles.inputIconActive : ""}`} onClick={toggleStt}>
                {sttOn ? <IconMic /> : <IconMicOff />}
              </button>
              <div className={styles.inputWrapper}>
                {isThinking && <div className={styles.thinkingBadge}><span /><span /><span /></div>}
                <input className={styles.inputField}
                  placeholder={isThinking ? "Generating sentence…" : "Type or sign a message…"}
                  value={inputText}
                  onChange={e => { setInputText(e.target.value); inputSourceRef.current = "manual"; }}
                  onKeyDown={e => { if (e.key === "Enter") sendMessage(); }} />
              </div>
              <button className={styles.inputIconBtn}><IconEmoji /></button>
              <button className={styles.sendBtn} onClick={sendMessage}><IconSend /></button>
            </div>
          </div>

          <div className={styles.avatarSpace}><div className={styles.avatarSpaceInner}><span className={styles.avatarSpaceLabel}>Avatar space</span></div></div>
        </div>
      </div>
    </div>
  );
}