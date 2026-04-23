import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams } from "react-router-dom";
import HomePage from "./pages/HomePage";
import CallRoom from "./pages/CallRoom";
import LearnASL from "./pages/LearnASL";
import LearnNumbersAlpha from "./pages/LearnNumbersAlpha";

function AppContent() {
  const [page, setPage] = useState("home");
  const navigate = useNavigate();
  const { roomCode } = useParams();

  useEffect(() => {
    if (roomCode) {
      setPage("call");
    }
  }, [roomCode]);

  function enterRoom(code, host) {
    navigate(`/call/${code}`);
  }

  function leaveRoom() {
    navigate("/");
    setPage("home");
  }

  return (
    <>
      {page === "home" && !roomCode && (
        <HomePage
          onEnterRoom={enterRoom}
          onLearnASL={() => setPage("learn-asl")}
          onLearnNumbers={() => setPage("learn-numbers")}
        />
      )}
      {(page === "call" || roomCode) && (
        <CallRoom roomCode={roomCode} isHost={false} onLeave={leaveRoom} />
      )}
      {page === "learn-asl" && (
        <LearnASL onBack={() => setPage("home")} />
      )}
      {page === "learn-numbers" && (
        <LearnNumbersAlpha onBack={() => setPage("home")} />
      )}
    </>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppContent />} />
        <Route path="/call/:roomCode" element={<AppContent />} />
      </Routes>
    </Router>
  );
}