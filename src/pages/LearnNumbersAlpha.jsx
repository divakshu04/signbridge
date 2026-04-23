import { useState } from "react";
import StarCanvas from "../components/StarCanvas";
import SignModal from "../components/SignModal";
import styles from "./LearnNumbersAlpha.module.css";

// ── Fingerspelling data ────────────────────────────────────────────────
// Each letter: name, fingers description, tip
const ALPHABET = [
  { char: "A", fingers: "Fist with thumb resting on side", tip: "Closed fist, thumb to the side — not over fingers" },
  { char: "B", fingers: "Four fingers up together, thumb tucked across palm", tip: "Fingers tall and straight, thumb folded in" },
  { char: "C", fingers: "Curved hand forming a C shape", tip: "Imagine holding a cup — that's the C shape" },
  { char: "D", fingers: "Index up, other fingers curl to touch thumb", tip: "The index points up while the rest form a circle" },
  { char: "E", fingers: "All fingers bent, tips touch thumb", tip: "Curl all fingers down like a claw touching the thumb" },
  { char: "F", fingers: "Index and thumb touch, other three fingers up", tip: "Make an OK sign but extend the three fingers straight" },
  { char: "G", fingers: "Index and thumb point sideways, parallel to ground", tip: "Like pointing a gun sideways — index and thumb out" },
  { char: "H", fingers: "Index and middle point sideways together", tip: "Two fingers pointing sideways like a flat gun" },
  { char: "I", fingers: "Pinky finger only extended, fist otherwise", tip: "Only the pinky up — everything else in a fist" },
  { char: "J", fingers: "Pinky up, draw a J shape in the air", tip: "Start like I, then arc down and hook left like the letter J" },
  { char: "K", fingers: "Index up, middle angled out, thumb between them", tip: "Index points up, middle points diagonally, thumb between" },
  { char: "L", fingers: "Index up, thumb out — L shape", tip: "Classic L shape — like the loser sign but serious" },
  { char: "M", fingers: "Three fingers folded over thumb", tip: "Tuck index, middle, ring over the thumb" },
  { char: "N", fingers: "Two fingers folded over thumb", tip: "Like M but only index and middle over the thumb" },
  { char: "O", fingers: "All fingers curve to meet thumb — O shape", tip: "Round all fingers and thumb to form a clear circle" },
  { char: "P", fingers: "Like K but hand points downward", tip: "Make K shape, then rotate wrist so fingers point down" },
  { char: "Q", fingers: "Like G but hand points downward", tip: "Make G shape, then point fingers toward the ground" },
  { char: "R", fingers: "Index and middle crossed", tip: "Cross your index over your middle finger" },
  { char: "S", fingers: "Fist with thumb over fingers", tip: "Like A but thumb rests over the front of the fingers" },
  { char: "T", fingers: "Thumb tucked between index and middle", tip: "Make a fist and push thumb between index and middle" },
  { char: "U", fingers: "Index and middle up together", tip: "Two fingers together pointing straight up" },
  { char: "V", fingers: "Index and middle up in a V shape", tip: "Peace sign — index and middle spread apart" },
  { char: "W", fingers: "Index, middle, and ring up, spread", tip: "Three fingers spread — like a wide peace sign" },
  { char: "X", fingers: "Index finger hooked/bent", tip: "Index finger curled like a hook" },
  { char: "Y", fingers: "Thumb and pinky out, others folded", tip: "Hang loose / shaka sign" },
  { char: "Z", fingers: "Index draws a Z shape in the air", tip: "Point index and draw Z in the air from left to right" },
];

const NUMBERS = [
  { num: "1",  fingers: "Index finger pointing up",                     tip: "One finger straight up — all others folded" },
  { num: "2",  fingers: "Index and middle up in V shape",               tip: "Peace sign — two fingers spread apart" },
  { num: "3",  fingers: "Thumb, index, middle extended",                tip: "Three fingers — thumb out plus the first two" },
  { num: "4",  fingers: "Four fingers up, thumb tucked",                tip: "All four fingers straight up, thumb folded in" },
  { num: "5",  fingers: "All five fingers spread open",                 tip: "Full open hand — all fingers wide" },
  { num: "6",  fingers: "Pinky and thumb touch, others extended",       tip: "Touch pinky to thumb, other three fingers up" },
  { num: "7",  fingers: "Ring finger and thumb touch, others extended", tip: "Touch ring finger to thumb, others up" },
  { num: "8",  fingers: "Middle finger and thumb touch, others up",     tip: "Touch middle finger to thumb, index and others up" },
  { num: "9",  fingers: "Index finger and thumb touch — O shape",       tip: "Make a small O with index and thumb, others up" },
  { num: "10", fingers: "Fist with thumb up, shake slightly",           tip: "Thumbs up fist, waggle the thumb side to side" },
];

// ── Letter card SVG — renders the letter visually ─────────────────────
function LetterCard({ char, fingers, tip, color, onClick }) {
  return (
    <div
      className={styles.card}
      onClick={() => onClick({ char, fingers, tip })}
      style={{ "--c": color }}
    >
      <div className={styles.cardFront}>
        <div className={styles.cardChar}>{char}</div>
        <div className={styles.cardHint}>tap for details</div>
      </div>
    </div>
  );
}

// ── Number card ───────────────────────────────────────────────────────
function NumberCard({ num, fingers, tip, color, onClick }) {
  return (
    <div
      className={`${styles.card} ${styles.cardWide}`}
      onClick={() => onClick({ num, fingers, tip })}
      style={{ "--c": color }}
    >
      <div className={styles.cardFront}>
        <div className={styles.cardNum}>{num}</div>
        <div className={styles.cardHint}>tap for details</div>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────
export default function LearnNumbersAlpha({ onBack }) {
  const [tab, setTab] = useState("alphabet"); // "alphabet" | "numbers"
  const [selectedSign, setSelectedSign] = useState(null);

  const alphaColors = [
    "#2663d9","#5a30c5","#0d7a60","#b45309","#7c3aed",
    "#065f46","#c2410c","#1e40af","#6d28d9","#047857",
    "#92400e","#9333ea","#0369a1","#15803d","#b91c1c",
    "#1d4ed8","#7e22ce","#0f766e","#a16207","#4338ca",
    "#0891b2","#16a34a","#dc2626","#7c3aed","#0284c7","#ca8a04",
  ];

  const numColors = [
    "#2663d9","#5a30c5","#0d7a60","#b45309","#7c3aed",
    "#065f46","#c2410c","#1e40af","#6d28d9","#047857",
  ];

  return (
    <div className={styles.root}>
      <StarCanvas />
      <div className={styles.aurora1} />
      <div className={styles.aurora2} />

      <div className={styles.page}>
        {/* Header */}
        <div className={styles.header}>
          <button className={styles.backBtn} onClick={onBack}>← Back</button>
          <div className={styles.headerCenter}>
            <h1 className={styles.pageTitle}>Numbers & Alphabet</h1>
            <p className={styles.pageSub}>ASL fingerspelling and number signs</p>
          </div>
          <div className={styles.headerRight} />
        </div>

        {/* Intro banner */}
        <div className={styles.banner}>
          <div className={styles.bannerIcon}>🤟</div>
          <div className={styles.bannerText}>
            <div className={styles.bannerTitle}>What is Fingerspelling?</div>
            <div className={styles.bannerSub}>
              Fingerspelling uses the 26 handshapes of the ASL manual alphabet to spell out words letter by letter. It is used for names, places, and words that have no ASL sign. Tap any card to see the hand description.
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${tab === "alphabet" ? styles.tabActive : ""}`}
            onClick={() => setTab("alphabet")}
          >
            A–Z Alphabet
          </button>
          <button
            className={`${styles.tab} ${tab === "numbers" ? styles.tabActive : ""}`}
            onClick={() => setTab("numbers")}
          >
            1–10 Numbers
          </button>
        </div>

        {/* Alphabet grid */}
        {tab === "alphabet" && (
          <>
            <div className={styles.sectionNote}>
              Tap a letter card to see the finger position and tip. Practice each letter slowly before building speed.
            </div>
            <div className={styles.alphaGrid}>
              {ALPHABET.map((item, i) => (
                <LetterCard
                  key={item.char}
                  {...item}
                  color={alphaColors[i % alphaColors.length]}
                  onClick={setSelectedSign}
                />
              ))}
            </div>
            <div className={styles.tipSection}>
              <div className={styles.tipSectionTitle}>💡 Fingerspelling Tips</div>
              <div className={styles.tipList}>
                <div className={styles.tipRow}>
                  <span className={styles.tipIcon}>🐢</span>
                  <span>Start slow — accuracy over speed. Speed comes naturally with practice.</span>
                </div>
                <div className={styles.tipRow}>
                  <span className={styles.tipIcon}>🪞</span>
                  <span>Practice in front of a mirror so you can see your own handshapes.</span>
                </div>
                <div className={styles.tipRow}>
                  <span className={styles.tipIcon}>✍️</span>
                  <span>Spell your own name every day — it builds muscle memory fast.</span>
                </div>
                <div className={styles.tipRow}>
                  <span className={styles.tipIcon}>👁️</span>
                  <span>When reading fingerspelling, focus on word shape — not individual letters.</span>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Numbers grid */}
        {tab === "numbers" && (
          <>
            <div className={styles.sectionNote}>
              ASL numbers 1–10 form the base for all larger numbers. Master these first before moving to 11–20.
            </div>
            <div className={styles.numGrid}>
              {NUMBERS.map((item, i) => (
                <NumberCard
                  key={item.num}
                  {...item}
                  color={numColors[i % numColors.length]}
                  onClick={setSelectedSign}
                />
              ))}
            </div>
            <div className={styles.tipSection}>
              <div className={styles.tipSectionTitle}>💡 Number Tips</div>
              <div className={styles.tipList}>
                <div className={styles.tipRow}>
                  <span className={styles.tipIcon}>🔢</span>
                  <span>Numbers 1–5 are straightforward. Numbers 6–9 use the thumb touching different fingers.</span>
                </div>
                <div className={styles.tipRow}>
                  <span className={styles.tipIcon}>🔄</span>
                  <span>Numbers above 10 combine signs — 11 is a flick of index, 12 flicks index and middle.</span>
                </div>
                <div className={styles.tipRow}>
                  <span className={styles.tipIcon}>📍</span>
                  <span>Keep numbers in neutral space in front of your body — not too high or low.</span>
                </div>
                <div className={styles.tipRow}>
                  <span className={styles.tipIcon}>🏪</span>
                  <span>Practice counting objects around you daily — very effective for retention.</span>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      <SignModal sign={selectedSign} onClose={() => setSelectedSign(null)} />
    </div>
  );
}