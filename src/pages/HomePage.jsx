import { useState } from "react";
import StarCanvas from "../components/StarCanvas";
import Modal from "../components/Modal";
import styles from "./HomePage.module.css";

/* ── SVG icons ── */
const IconVideo = () => (
  <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
    <path d="M15 10l4.553-2.368A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/>
  </svg>
);
const IconLink = () => (
  <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
    <path d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
  </svg>
);
const IconBook = () => (
  <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
    <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
  </svg>
);

/* ── City silhouette ── */
function CitySkyline() {
  return (
    <svg className={styles.city} xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 1440 220" preserveAspectRatio="xMidYMax slice">
      <defs>
        <linearGradient id="city-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#1a1050" stopOpacity="0.9"/>
          <stop offset="100%" stopColor="#0a0820" stopOpacity="1"/>
        </linearGradient>
      </defs>
      <path fill="url(#city-grad)" opacity="0.6"
        d="M0,220 L0,160 L40,160 L40,140 L60,140 L60,120 L80,120 L80,100 L100,100 L100,80 L120,80 L120,100 L140,100 L140,60 L160,60 L160,100 L180,100 L180,80 L200,80 L200,140 L240,140 L240,120 L260,120 L260,100 L280,100 L280,80 L300,80 L300,60 L320,60 L320,80 L340,80 L340,100 L360,100 L360,120 L380,120 L380,80 L400,80 L400,60 L420,60 L420,40 L440,40 L440,60 L460,60 L460,80 L480,80 L480,100 L520,100 L520,80 L540,80 L540,60 L560,60 L560,80 L580,80 L580,100 L600,100 L600,80 L620,80 L620,60 L640,60 L640,80 L680,80 L680,60 L700,60 L700,40 L720,40 L720,60 L740,60 L740,80 L760,80 L760,100 L800,100 L800,80 L820,80 L820,60 L840,60 L840,40 L860,40 L860,60 L880,60 L880,80 L900,80 L900,100 L940,100 L940,120 L960,120 L960,100 L980,100 L980,80 L1000,80 L1000,60 L1020,60 L1020,80 L1060,80 L1060,100 L1080,100 L1080,80 L1100,80 L1100,60 L1120,60 L1120,40 L1140,40 L1140,60 L1160,60 L1160,80 L1200,80 L1200,100 L1220,100 L1220,80 L1240,80 L1240,60 L1260,60 L1260,80 L1280,80 L1280,100 L1300,100 L1300,80 L1320,80 L1320,60 L1340,60 L1340,80 L1360,80 L1360,100 L1380,100 L1380,140 L1440,140 L1440,220 Z"/>
      <path fill="#0d0a28"
        d="M0,220 L0,180 L50,180 L50,160 L70,160 L70,155 L90,155 L90,160 L110,160 L110,175 L130,175 L130,155 L150,155 L150,140 L170,140 L170,155 L190,155 L190,175 L210,175 L210,160 L230,160 L230,145 L250,145 L250,160 L270,160 L270,175 L290,175 L290,155 L310,155 L310,140 L330,140 L330,120 L350,120 L350,140 L370,140 L370,155 L390,155 L390,170 L420,170 L420,155 L440,155 L440,140 L460,140 L460,120 L480,120 L480,140 L500,140 L500,155 L530,155 L530,170 L560,170 L560,155 L580,155 L580,140 L600,140 L600,120 L620,120 L620,100 L640,100 L640,120 L660,120 L660,140 L680,140 L680,155 L700,155 L700,170 L730,170 L730,155 L750,155 L750,135 L770,135 L770,120 L790,120 L790,135 L810,135 L810,155 L840,155 L840,170 L870,170 L870,155 L890,155 L890,140 L910,140 L910,120 L930,120 L930,140 L950,140 L950,155 L980,155 L980,170 L1010,170 L1010,155 L1030,155 L1030,140 L1050,140 L1050,120 L1070,120 L1070,100 L1090,100 L1090,120 L1110,120 L1110,140 L1130,140 L1130,160 L1160,160 L1160,175 L1190,175 L1190,160 L1210,160 L1210,145 L1230,145 L1230,160 L1250,160 L1250,175 L1280,175 L1280,160 L1300,160 L1300,145 L1320,145 L1320,160 L1340,160 L1340,175 L1370,175 L1370,180 L1440,180 L1440,220 Z"/>
      <g fill="#ffe8a0" opacity="0.5">
        <rect x="155" y="148" width="3" height="4" rx="1"/>
        <rect x="163" y="148" width="3" height="4" rx="1"/>
        <rect x="334" y="128" width="3" height="4" rx="1"/>
        <rect x="342" y="135" width="3" height="4" rx="1"/>
        <rect x="463" y="128" width="3" height="4" rx="1"/>
        <rect x="623" y="108" width="3" height="4" rx="1"/>
        <rect x="773" y="128" width="3" height="4" rx="1"/>
        <rect x="1073" y="108" width="3" height="4" rx="1"/>
        <rect x="1093" y="122" width="3" height="4" rx="1"/>
      </g>
      <rect x="0" y="175" width="1440" height="45" fill="url(#city-grad)" opacity="0.85"/>
    </svg>
  );
}

/* ── SignBridge logo SVG ── */
function LogoIcon() {
  return (
    <svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg" width="72" height="72">
      <defs>
        <linearGradient id="arc1" x1="0" y1="0" x2="80" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#5a8ef0"/>
          <stop offset="100%" stopColor="#c0a0ff"/>
        </linearGradient>
        <linearGradient id="arc2" x1="0" y1="0" x2="80" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#3b7de8"/>
          <stop offset="100%" stopColor="#9f70f8"/>
        </linearGradient>
      </defs>
      <path d="M10 55 Q40 8 70 55"  stroke="url(#arc1)" strokeWidth="5" strokeLinecap="round" fill="none"/>
      <path d="M20 55 Q40 20 60 55" stroke="url(#arc2)" strokeWidth="4" strokeLinecap="round" fill="none" opacity="0.7"/>
      <line x1="10" y1="55" x2="10" y2="66" stroke="url(#arc1)" strokeWidth="4" strokeLinecap="round"/>
      <line x1="70" y1="55" x2="70" y2="66" stroke="url(#arc1)" strokeWidth="4" strokeLinecap="round"/>
      <line x1="4"  y1="66" x2="76" y2="66" stroke="url(#arc1)" strokeWidth="3" strokeLinecap="round"/>
    </svg>
  );
}

/* ── Main page ── */
export default function HomePage({ onEnterRoom, onLearnASL, onLearnNumbers }) {
  const [modal, setModal] = useState(null); // null | string

  const cards = [
    { id: "meet",  label: "Start a Meeting",    icon: <IconVideo/>, cls: styles.cardMeet  },
    { id: "join",  label: "Join a Meeting",     icon: <IconLink/>,  cls: styles.cardJoin  },
    { id: "learn", label: "Learn Sign Language",icon: <IconBook/>,  cls: styles.cardLearn },
  ];

  const features = [
    { id: "sign2text",  emoji: "🤚", label: "Sign to Text"   },
    { id: "voice2sign", emoji: "🎙️", label: "Voice to Signs" },
    { id: "practice",   emoji: "📖", label: "Learn & Practice"},
  ];

  return (
    <>
      <StarCanvas />

      {/* Aurora glows */}
      <div className={`${styles.aurora} ${styles.aurora1}`} />
      <div className={`${styles.aurora} ${styles.aurora2}`} />
      <div className={`${styles.aurora} ${styles.aurora3}`} />

      <CitySkyline />

      <main className={styles.page}>
        {/* Logo */}
        <div className={styles.logoWrap}>
          <LogoIcon />
          <span className={styles.logoName}>SignBridge</span>
          <span className={styles.logoTagline}>Connect Beyond Words</span>
        </div>

        {/* Hero */}
        <div className={styles.heroText}>
          <h1 className={styles.heroTitle}>Welcome to SignBridge</h1>
          <p className={styles.heroSub}>Bridging Communication for Everyone</p>
        </div>

        {/* Cards */}
        <div className={styles.cards}>
          {cards.map((c) => (
            <button key={c.id} className={`${styles.card} ${c.cls}`}
              onClick={() => setModal(c.id)} aria-label={c.label}>
              <div className={styles.cardIcon}>{c.icon}</div>
              <span className={styles.cardLabel}>{c.label}</span>
            </button>
          ))}
        </div>
      </main>

      {/* Feature bar */}
      <nav className={styles.featureBar}>
        {features.map((f, i) => (
          <div key={f.id} style={{ display: "flex", alignItems: "center", gap: "inherit" }}>
            {i > 0 && <div className={styles.featSep} />}
            <button className={styles.feat} onClick={() => setModal(f.id)}>
              <span className={styles.featIcon}>{f.emoji}</span>
              <span className={styles.featLabel}>{f.label}</span>
            </button>
          </div>
        ))}
      </nav>

      {/* Modal */}
      {modal && (
        <Modal
          type={modal}
          onClose={() => setModal(null)}
          onEnterRoom={(code) => {
            // Explicitly determine role based on which modal is open
            const isHosting = modal === "meet";
            setModal(null);
            onEnterRoom(code, isHosting);
          }}
          onNavigate={(page) => {
            setModal(null);
            if (page === "learn-asl") onLearnASL?.(); else if (page === "learn-numbers") onLearnNumbers?.();
          }}
        />
      )}
    </>
  );
}