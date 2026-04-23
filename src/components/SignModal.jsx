import { useEffect } from "react";
import styles from "./SignModal.module.css";

export default function SignModal({ sign, onClose }) {
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  if (!sign) return null;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.box} onClick={(e) => e.stopPropagation()}>
        <button className={styles.close} onClick={onClose}>×</button>
        <div className={styles.content}>
          <div className={styles.signChar}>{sign.char || sign.num}</div>
          <div className={styles.signFingers}>{sign.fingers}</div>
          <div className={styles.signTip}>💡 {sign.tip}</div>
        </div>
      </div>
    </div>
  );
}