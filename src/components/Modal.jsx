import { useEffect, useRef } from "react";
import styles from "./Modal.module.css";

export default function Modal({ type, onClose, onEnterRoom, onNavigate }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const roomCodeRef = useRef("SB-" + Math.floor(1000 + Math.random() * 9000));
  const roomCode = roomCodeRef.current;

  function copyCode() {
    navigator.clipboard.writeText(roomCode).then(() => {
      const btn = document.getElementById("copy-btn");
      if (!btn) return;
      btn.textContent = "Copied!";
      btn.style.color = "#4ecda4";
      setTimeout(() => { btn.textContent = "Copy"; btn.style.color = ""; }, 2000);
    });
  }

  function handleEnterRoom() {
    if (onEnterRoom) onEnterRoom(roomCode, true); // true = host
    onClose();
  }

  function joinRoom() {
    const input = document.getElementById("join-input");
    const code  = input?.value.trim();
    if (!code) { if (input) input.style.borderColor = "#e85d60"; return; }
    if (onEnterRoom) onEnterRoom(code, false); // false = guest
    onClose();
  }

  const content = {
    meet: (
      <>
        <h2 className={styles.title}>Start a Meeting</h2>
        <p className={styles.sub}>Share this code with the person you want to connect with.</p>
        <div className={styles.codeBox}>
          <div className={styles.codeLabel}>Your room code</div>
          <div className={styles.codeRow}>
            <span id="room-code-val" className={styles.codeVal}>{roomCode}</span>
            <button id="copy-btn" className={styles.copyBtn} onClick={copyCode}>Copy</button>
          </div>
        </div>
        <button className={`${styles.btn} ${styles.btnBlue}`} onClick={handleEnterRoom}>
          Enter Room
        </button>
      </>
    ),
    join: (
      <>
        <h2 className={styles.title}>Join a Meeting</h2>
        <p className={styles.sub}>Enter the room code shared by the other person.</p>
        <input id="join-input" className={styles.input}
          placeholder="e.g. SB-4829" maxLength={10} autoComplete="off"/>
        <button className={`${styles.btn} ${styles.btnViolet}`} onClick={joinRoom}>Join Room</button>
      </>
    ),
    learn: (
      <>
        <h2 className={styles.title}>Learn Sign Language</h2>
        <p className={styles.sub}>Choose a learning path to get started.</p>
        <button className={`${styles.btn} ${styles.btnBlue}`} style={{marginBottom:12}}
          onClick={() => { onClose(); onNavigate?.("learn-asl"); }}>ASL Basics</button>
        <button className={`${styles.btn} ${styles.btnViolet}`} style={{marginBottom:12}}
          onClick={() => { onClose(); onNavigate?.("learn-numbers"); }}>Numbers and Alphabet</button>
        <button className={`${styles.btn} ${styles.btnTeal}`}
          onClick={() => alert("Everyday phrases coming soon!")}>Everyday Phrases</button>
      </>
    ),
    sign2text: (
      <>
        <h2 className={styles.title}>Sign to Text</h2>
        <p className={styles.sub}>Use your camera to translate hand signs into written text in real time.</p>
        <button className={`${styles.btn} ${styles.btnBlue}`}
          onClick={() => alert("Coming in next build!")}>Open Camera</button>
      </>
    ),
    voice2sign: (
      <>
        <h2 className={styles.title}>Voice to Signs</h2>
        <p className={styles.sub}>Speak and watch it transform into sign language animations.</p>
        <button className={`${styles.btn} ${styles.btnViolet}`}
          onClick={() => alert("Coming in next build!")}>Start Speaking</button>
      </>
    ),
    practice: (
      <>
        <h2 className={styles.title}>Learn and Practice</h2>
        <p className={styles.sub}>Interactive lessons to master sign language at your own pace.</p>
        <button className={`${styles.btn} ${styles.btnTeal}`}
          onClick={() => alert("Coming soon!")}>Start Practicing</button>
      </>
    ),
  };

  return (
    <div className={styles.overlay}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className={styles.box}>
        <button className={styles.close} onClick={onClose} aria-label="Close">X</button>
        {content[type]}
      </div>
    </div>
  );
}