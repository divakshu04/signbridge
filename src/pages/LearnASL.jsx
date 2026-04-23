import { useState } from "react";
import StarCanvas from "../components/StarCanvas";
import styles from "./LearnASL.module.css";

// ── Hand shape SVG illustrations ──────────────────────────────────────
const HandSVG = ({ type }) => {
  const shapes = {
    open: (
      <svg viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="30" y="2" width="10" height="35" rx="5" fill="#a0c4ff" opacity="0.9"/>
        <rect x="18" y="8" width="10" height="33" rx="5" fill="#a0c4ff" opacity="0.9"/>
        <rect x="42" y="6" width="10" height="34" rx="5" fill="#a0c4ff" opacity="0.9"/>
        <rect x="54" y="10" width="10" height="30" rx="5" fill="#a0c4ff" opacity="0.9"/>
        <rect x="8" y="22" width="9" height="24" rx="4" fill="#a0c4ff" opacity="0.9"/>
        <rect x="16" y="38" width="50" height="50" rx="10" fill="#5a8ef0" opacity="0.8"/>
      </svg>
    ),
    fist: (
      <svg viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="15" y="30" width="50" height="55" rx="12" fill="#5a8ef0" opacity="0.9"/>
        <rect x="15" y="28" width="12" height="20" rx="6" fill="#4c7de0" opacity="0.9"/>
        <rect x="29" y="26" width="12" height="22" rx="6" fill="#4c7de0" opacity="0.9"/>
        <rect x="43" y="28" width="12" height="20" rx="6" fill="#4c7de0" opacity="0.9"/>
        <rect x="57" y="30" width="10" height="18" rx="5" fill="#4c7de0" opacity="0.9"/>
        <rect x="5" y="40" width="14" height="22" rx="7" fill="#5a8ef0" opacity="0.85"/>
      </svg>
    ),
    point: (
      <svg viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="15" y="35" width="50" height="55" rx="12" fill="#5a8ef0" opacity="0.9"/>
        <rect x="33" y="4" width="11" height="40" rx="5.5" fill="#a0c4ff" opacity="0.95"/>
        <rect x="15" y="33" width="12" height="18" rx="6" fill="#4c7de0" opacity="0.9"/>
        <rect x="43" y="33" width="12" height="18" rx="6" fill="#4c7de0" opacity="0.9"/>
        <rect x="57" y="35" width="10" height="16" rx="5" fill="#4c7de0" opacity="0.9"/>
        <rect x="5" y="44" width="14" height="22" rx="7" fill="#5a8ef0" opacity="0.85"/>
      </svg>
    ),
    two: (
      <svg viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="15" y="38" width="50" height="52" rx="12" fill="#5a8ef0" opacity="0.9"/>
        <rect x="25" y="5" width="11" height="40" rx="5.5" fill="#a0c4ff" opacity="0.95"/>
        <rect x="40" y="5" width="11" height="40" rx="5.5" fill="#a0c4ff" opacity="0.95"/>
        <rect x="53" y="36" width="10" height="18" rx="5" fill="#4c7de0" opacity="0.9"/>
        <rect x="5" y="46" width="14" height="22" rx="7" fill="#5a8ef0" opacity="0.85"/>
      </svg>
    ),
    wave: (
      <svg viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="30" y="2" width="10" height="35" rx="5" fill="#a0c4ff" opacity="0.9" transform="rotate(-10 35 20)"/>
        <rect x="18" y="8" width="10" height="33" rx="5" fill="#a0c4ff" opacity="0.9" transform="rotate(-8 23 25)"/>
        <rect x="42" y="6" width="10" height="34" rx="5" fill="#a0c4ff" opacity="0.9" transform="rotate(-12 47 23)"/>
        <rect x="54" y="10" width="10" height="30" rx="5" fill="#a0c4ff" opacity="0.9" transform="rotate(-14 59 25)"/>
        <rect x="8" y="22" width="9" height="24" rx="4" fill="#a0c4ff" opacity="0.9"/>
        <rect x="16" y="38" width="50" height="50" rx="10" fill="#5a8ef0" opacity="0.8"/>
        <path d="M5 55 Q20 45 35 55 Q50 65 65 55" stroke="#4ecda4" strokeWidth="2.5" strokeLinecap="round" fill="none" opacity="0.7"/>
      </svg>
    ),
    cshape: (
      <svg viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M55 20 Q65 35 65 50 Q65 65 55 78 Q45 85 32 83 Q18 80 12 68 Q6 55 10 42 Q15 28 28 22 Q40 16 55 20Z" 
          fill="none" stroke="#5a8ef0" strokeWidth="12" strokeLinecap="round"
          strokeDasharray="120 50"/>
        <path d="M55 20 Q65 35 65 50 Q65 65 55 78" 
          fill="none" stroke="#a0c4ff" strokeWidth="8" strokeLinecap="round" opacity="0.6"/>
      </svg>
    ),
    thumbsup: (
      <svg viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="20" y="40" width="45" height="50" rx="10" fill="#5a8ef0" opacity="0.9"/>
        <rect x="30" y="10" width="13" height="38" rx="6.5" fill="#a0c4ff" opacity="0.95" transform="rotate(-15 36 29)"/>
        <rect x="20" y="38" width="12" height="18" rx="6" fill="#4c7de0" opacity="0.9"/>
        <rect x="34" y="36" width="12" height="18" rx="6" fill="#4c7de0" opacity="0.9"/>
        <rect x="48" y="38" width="10" height="16" rx="5" fill="#4c7de0" opacity="0.9"/>
        <rect x="10" y="46" width="14" height="24" rx="7" fill="#5a8ef0" opacity="0.85"/>
      </svg>
    ),
    flat: (
      <svg viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="35" width="60" height="12" rx="6" fill="#a0c4ff" opacity="0.95"/>
        <rect x="10" y="22" width="11" height="25" rx="5.5" fill="#a0c4ff" opacity="0.9"/>
        <rect x="23" y="20" width="11" height="27" rx="5.5" fill="#a0c4ff" opacity="0.9"/>
        <rect x="36" y="20" width="11" height="27" rx="5.5" fill="#a0c4ff" opacity="0.9"/>
        <rect x="49" y="22" width="11" height="25" rx="5.5" fill="#a0c4ff" opacity="0.9"/>
        <rect x="8" y="42" width="12" height="18" rx="6" fill="#5a8ef0" opacity="0.85" transform="rotate(-30 14 51)"/>
        <rect x="10" y="44" width="62" height="40" rx="10" fill="#5a8ef0" opacity="0.8"/>
      </svg>
    ),
  };
  return (
    <div className={styles.handSvg}>
      {shapes[type] || shapes.open}
    </div>
  );
};

// ── Lesson data ───────────────────────────────────────────────────────
const LESSONS = [
  {
    id: 1,
    title: "What is ASL?",
    emoji: "🌟",
    color: "#2663d9",
    duration: "3 min read",
    type: "intro",
    content: {
      heading: "American Sign Language",
      intro: "ASL is a complete, natural language that uses hand shapes, movement, facial expressions, and body posture to communicate. It is the primary language of Deaf communities in the United States and Canada.",
      facts: [
        { icon: "👥", text: "Used by over 500,000 people in the US as their primary language" },
        { icon: "🧠", text: "ASL has its own grammar — different from English" },
        { icon: "👁️", text: "Facial expressions carry grammatical meaning, not just emotion" },
        { icon: "✋", text: "Both hands work together — dominant and non-dominant" },
        { icon: "📍", text: "Location, movement, and handshape all change meaning" },
      ],
      tip: "Learning ASL opens communication with the Deaf community and builds empathy and awareness.",
    },
  },
  {
    id: 2,
    title: "Basic Handshapes",
    emoji: "✋",
    color: "#5a30c5",
    duration: "5 min",
    type: "handshapes",
    content: {
      heading: "The 5 Core Handshapes",
      intro: "All signs are built from combinations of these fundamental handshapes. Mastering them is the first step.",
      shapes: [
        { name: "Open Hand", svg: "open", desc: "All 5 fingers extended and spread apart. Used in signs like HELLO, PLEASE, and HAPPY." },
        { name: "Flat Hand", svg: "flat", desc: "All fingers together, palm flat. Used in signs like THANK YOU and many directional signs." },
        { name: "Fist (A shape)", svg: "fist", desc: "All fingers curled into palm, thumb on side. Used in signs like YES." },
        { name: "Index Point", svg: "point", desc: "Only index finger extended. Used for pointing, directions, and signs like THINK, GO." },
        { name: "V / Peace shape", svg: "two", desc: "Index and middle fingers extended, others curled. Used in signs like LOOK and SEE." },
      ],
    },
  },
  {
    id: 3,
    title: "Greetings",
    emoji: "👋",
    color: "#0d7a60",
    duration: "5 min",
    type: "signs",
    content: {
      heading: "Greetings in ASL",
      intro: "Start every conversation with confidence. These signs are used every day.",
      signs: [
        {
          word: "HELLO",
          svg: "wave",
          steps: [
            "Hold your dominant hand at forehead level",
            "All fingers extended, palm facing outward",
            "Move hand outward to the side in a smooth wave",
          ],
          tip: "Think of a salute that turns into a wave.",
          video: "https://www.youtube.com/embed/FVjpLa8GqeM",
        },
        {
          word: "BYE",
          svg: "wave",
          steps: [
            "Hold open hand at ear/shoulder level",
            "Fingers extended, palm facing outward",
            "Wave side to side naturally like waving goodbye",
          ],
          tip: "Similar to HELLO but at a lower position — ear level instead of forehead.",
          video: "https://www.youtube.com/embed/4e14uNAn2Ao",
        },
        {
          word: "THANK YOU",
          svg: "flat",
          steps: [
            "Start with flat hand touching your chin",
            "Move hand forward and slightly down",
            "End with palm facing up toward the other person",
          ],
          tip: "Imagine blowing a thankful kiss from your chin outward.",
          video: "https://www.youtube.com/embed/IvRwNLNR4_w",
        },
        {
          word: "PLEASE",
          svg: "open",
          steps: [
            "Place flat open hand on your chest",
            "Make a circular rubbing motion clockwise",
            "Repeat 2 times with a polite expression",
          ],
          tip: "The circular motion on the chest shows sincerity and politeness.",
          video: "https://www.youtube.com/embed/rnb9FxPO7is",
        },
      ],
    },
  },
  {
    id: 4,
    title: "Yes & No",
    emoji: "👍",
    color: "#b45309",
    duration: "4 min",
    type: "signs",
    content: {
      heading: "Agreement & Disagreement",
      intro: "Two of the most important signs in any conversation.",
      signs: [
        {
          word: "YES",
          svg: "fist",
          steps: [
            "Make a fist with your dominant hand",
            "Hold it at chest or neutral level, palm facing outward",
            "Nod the fist up and down like a head nodding yes",
          ],
          tip: "The fist mimics a head nodding in agreement. Keep wrist loose.",
          video: "https://www.youtube.com/embed/0usayvOXzHo",
        },
        {
          word: "NO",
          svg: "two",
          steps: [
            "Extend your index and middle fingers",
            "Hold at neutral level in front of you",
            "Quickly bring fingers to thumb like a snap, twice",
          ],
          tip: "The two fingers closing represents shutting something down.",
          video: "https://www.youtube.com/embed/QJXKaOSyl4o",
        },
      ],
    },
  },
  {
    id: 5,
    title: "Feelings",
    emoji: "😊",
    color: "#7c3aed",
    duration: "6 min",
    type: "signs",
    content: {
      heading: "Expressing Feelings",
      intro: "Communicating emotions is essential. Facial expression is just as important as the hand sign.",
      signs: [
        {
          word: "HAPPY",
          svg: "open",
          steps: [
            "Place flat open hand on chest",
            "Brush upward in a quick motion — once or twice",
            "Smile naturally while doing this",
          ],
          tip: "The upward brush represents lifting your spirits. Always smile!",
          video: "https://www.youtube.com/embed/N5GLqFNS3Uo",
        },
        {
          word: "SAD",
          svg: "open",
          steps: [
            "Hold both open hands in front of your face, palms toward you",
            "Slowly move both hands downward",
            "Let your face show sadness — eyebrows down",
          ],
          tip: "The face is half the sign. Sad expression makes it real.",
          video: "https://www.youtube.com/embed/zfkQYQhrZ6Q",
        },
        {
          word: "HUNGRY",
          svg: "cshape",
          steps: [
            "Form a C-shape with your dominant hand",
            "Place it on your upper chest near your throat",
            "Slide hand down your chest to your stomach",
          ],
          tip: "The C-hand moving down the body represents an empty stomach.",
          video: "https://www.youtube.com/embed/8ZOUoDZkAoQ",
        },
      ],
    },
  },
  {
    id: 6,
    title: "Facial Expressions",
    emoji: "😄",
    color: "#c2410c",
    duration: "4 min",
    type: "intro",
    content: {
      heading: "Why Facial Expressions Matter",
      intro: "In ASL, facial expressions are not optional — they are grammatically required. They change the meaning of signs completely.",
      facts: [
        { icon: "🤨", text: "Raised eyebrows = Yes/No question. e.g. 'Are you hungry?' — eyebrows up" },
        { icon: "😟", text: "Furrowed brows = Wh-question. e.g. 'Where are you going?' — brows down" },
        { icon: "😊", text: "Positive expression + HAPPY = stronger meaning of happiness" },
        { icon: "😢", text: "Sad face without SAD sign still communicates sadness to a Deaf person" },
        { icon: "😲", text: "Mouth shapes can add meaning — puffed cheeks = large/heavy" },
      ],
      tip: "Practice in front of a mirror. If your face looks natural while signing, you are doing it right.",
    },
  },
  {
    id: 7,
    title: "Tips for Learning",
    emoji: "💡",
    color: "#065f46",
    duration: "3 min read",
    type: "tips",
    content: {
      heading: "How to Learn ASL Effectively",
      tips: [
        { icon: "🪞", title: "Practice in a mirror", desc: "Watch your handshapes, facial expressions, and body position. This is how Deaf signers self-correct." },
        { icon: "🔁", title: "Repeat consistently", desc: "Short 10-minute daily sessions are far better than one long session per week. Muscle memory builds slowly." },
        { icon: "👁️", title: "Watch native signers", desc: "YouTube and resources like ASL University (lifeprint.com) show real, natural ASL from native signers." },
        { icon: "🤝", title: "Interact with Deaf people", desc: "The best way to learn any language is to use it. Attend local Deaf events or join online ASL communities." },
        { icon: "📱", title: "Use SignBridge", desc: "Practice signing in the Sign to Text mode. Our AI model will give you real-time feedback on your signs." },
        { icon: "🎯", title: "Learn concepts not words", desc: "ASL is not English on your hands. Think in concepts and visual ideas, not word-for-word translation." },
      ],
    },
  },
];

// ── Lesson card component ─────────────────────────────────────────────
function LessonCard({ lesson, index, onOpen, completed }) {
  return (
    <button
      className={`${styles.lessonCard} ${completed ? styles.lessonDone : ""}`}
      onClick={() => onOpen(lesson)}
      style={{ "--card-color": lesson.color, animationDelay: `${index * 0.08}s` }}
    >
      <div className={styles.lessonNum}>{String(index + 1).padStart(2, "0")}</div>
      <div className={styles.lessonEmoji}>{lesson.emoji}</div>
      <div className={styles.lessonInfo}>
        <div className={styles.lessonTitle}>{lesson.title}</div>
        <div className={styles.lessonMeta}>
          <span className={styles.lessonDuration}>{lesson.duration}</span>
          {completed && <span className={styles.lessonBadge}>✓ Done</span>}
        </div>
      </div>
      <div className={styles.lessonArrow}>›</div>
      {completed && <div className={styles.lessonDoneLine} />}
    </button>
  );
}

// ── Sign card within lesson ───────────────────────────────────────────
function SignCard({ sign, index }) {
  const [open, setOpen] = useState(index === 0);
  return (
    <div className={`${styles.signCard} ${open ? styles.signCardOpen : ""}`}>
      <button className={styles.signCardHeader} onClick={() => setOpen(!open)}>
        <div className={styles.signName}>{sign.word}</div>
        <div className={styles.signChevron}>{open ? "▲" : "▼"}</div>
      </button>
      {open && (
        <div className={styles.signCardBody}>
          <div className={styles.signVisual}>
            <HandSVG type={sign.svg} />
          </div>
          <div className={styles.signSteps}>
            {sign.steps.map((step, i) => (
              <div key={i} className={styles.signStep}>
                <div className={styles.stepNum}>{i + 1}</div>
                <div className={styles.stepText}>{step}</div>
              </div>
            ))}
            <div className={styles.signTip}>
              <span>💡</span> {sign.tip}
            </div>
            {sign.video && (
              <div className={styles.signVideo}>
                <iframe
                  width="100%"
                  height="315"
                  src={sign.video}
                  title={`${sign.word} sign demonstration`}
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  allowFullScreen
                  loading="lazy"
                ></iframe>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Lesson viewer ─────────────────────────────────────────────────────
function LessonViewer({ lesson, onClose, onComplete, completed, lessons, onNavigate }) {
  const currentIdx = lessons.findIndex(l => l.id === lesson.id);

  return (
    <div className={styles.lessonViewer}>
      <div className={styles.viewerHeader}>
        <button className={styles.viewerBack} onClick={onClose}>← Back</button>
        <div className={styles.viewerProgress}>
          Lesson {currentIdx + 1} of {lessons.length}
        </div>
        {!completed && (
          <button className={styles.viewerComplete} onClick={() => onComplete(lesson.id)}>
            Mark Done ✓
          </button>
        )}
        {completed && <span className={styles.viewerDone}>✓ Completed</span>}
      </div>

      <div className={styles.viewerBody}>
        <div className={styles.viewerEmoji}>{lesson.emoji}</div>
        <h2 className={styles.viewerTitle}>{lesson.content.heading}</h2>
        <p className={styles.viewerIntro}>{lesson.content.intro}</p>

        {/* Facts layout */}
        {lesson.content.facts && (
          <div className={styles.factsList}>
            {lesson.content.facts.map((fact, i) => (
              <div key={i} className={styles.factItem}>
                <span className={styles.factIcon}>{fact.icon}</span>
                <span className={styles.factText}>{fact.text}</span>
              </div>
            ))}
            <div className={styles.tipBox}>
              <span>💡</span> {lesson.content.tip}
            </div>
          </div>
        )}

        {/* Handshapes layout */}
        {lesson.content.shapes && (
          <div className={styles.shapesList}>
            {lesson.content.shapes.map((shape, i) => (
              <div key={i} className={styles.shapeItem}>
                <HandSVG type={shape.svg} />
                <div className={styles.shapeInfo}>
                  <div className={styles.shapeName}>{shape.name}</div>
                  <div className={styles.shapeDesc}>{shape.desc}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Signs layout */}
        {lesson.content.signs && (
          <div className={styles.signsList}>
            {lesson.content.signs.map((sign, i) => (
              <SignCard key={i} sign={sign} index={i} />
            ))}
          </div>
        )}

        {/* Tips layout */}
        {lesson.content.tips && (
          <div className={styles.tipsList}>
            {lesson.content.tips.map((tip, i) => (
              <div key={i} className={styles.tipItem}>
                <div className={styles.tipIcon}>{tip.icon}</div>
                <div className={styles.tipContent}>
                  <div className={styles.tipTitle}>{tip.title}</div>
                  <div className={styles.tipDesc}>{tip.desc}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className={styles.viewerNav}>
        <button
          className={styles.navBtn}
          disabled={currentIdx === 0}
          onClick={() => onNavigate(lessons[currentIdx - 1])}
        >
          ← Previous
        </button>
        <button
          className={`${styles.navBtn} ${styles.navBtnNext}`}
          disabled={currentIdx === lessons.length - 1}
          onClick={() => onNavigate(lessons[currentIdx + 1])}
        >
          Next →
        </button>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────
export default function LearnASL({ onBack }) {
  const [activeLesson, setActiveLesson] = useState(null);
  const [completed,    setCompleted]    = useState(new Set());

  function markDone(id) {
    setCompleted(prev => new Set([...prev, id]));
  }

  const progress = Math.round((completed.size / LESSONS.length) * 100);

  return (
    <div className={styles.root}>
      <StarCanvas />
      <div className={styles.aurora1} />
      <div className={styles.aurora2} />

      {activeLesson ? (
        <LessonViewer
          lesson={activeLesson}
          lessons={LESSONS}
          completed={completed.has(activeLesson.id)}
          onClose={() => setActiveLesson(null)}
          onComplete={(id) => markDone(id)}
          onNavigate={(lesson) => setActiveLesson(lesson)}
        />
      ) : (
        <div className={styles.page}>
          {/* Header */}
          <div className={styles.header}>
            <button className={styles.backBtn} onClick={onBack}>← Back</button>
            <div className={styles.headerCenter}>
              <h1 className={styles.pageTitle}>ASL Basics</h1>
              <p className={styles.pageSub}>Learn American Sign Language step by step</p>
            </div>
            <div className={styles.headerRight} />
          </div>

          {/* Progress bar */}
          <div className={styles.progressWrap}>
            <div className={styles.progressBar}>
              <div className={styles.progressFill} style={{ width: `${progress}%` }} />
            </div>
            <div className={styles.progressLabel}>
              {completed.size}/{LESSONS.length} lessons completed — {progress}%
            </div>
          </div>

          {/* Intro banner */}
          <div className={styles.banner}>
            <div className={styles.bannerIcon}>🤟</div>
            <div className={styles.bannerText}>
              <div className={styles.bannerTitle}>Welcome to ASL Basics</div>
              <div className={styles.bannerSub}>
                7 lessons covering handshapes, greetings, feelings, and signing tips. Complete each lesson at your own pace.
              </div>
            </div>
          </div>

          {/* Lessons list */}
          <div className={styles.lessonsList}>
            {LESSONS.map((lesson, i) => (
              <LessonCard
                key={lesson.id}
                lesson={lesson}
                index={i}
                completed={completed.has(lesson.id)}
                onOpen={setActiveLesson}
              />
            ))}
          </div>

          {/* Footer note */}
          <div className={styles.footerNote}>
            After completing these lessons, try signing in a video call to practice in real time.
          </div>
        </div>
      )}
    </div>
  );
}