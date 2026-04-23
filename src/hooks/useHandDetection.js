import { useEffect, useRef, useState, useCallback } from "react";

export function useHandDetection(videoRef, isActive, onWordDetected, getContextFunc = null) {
  const canvasRef   = useRef(null);
  const holisticRef = useRef(null);
  const rafRef      = useRef(null);
  const wsRef       = useRef(null);

  const [handsDetected, setHandsDetected] = useState(false);
  const [wsConnected,   setWsConnected]   = useState(false);
  const [debugInfo,     setDebugInfo]     = useState(null);

  // ── WebSocket to Python server ──────────────────────────────────
  useEffect(() => {
    function connect() {
      const ws = new WebSocket(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/ws/signs`);

      ws.onopen = () => {
        console.log("✓ Connected to SignBridge server");
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const result = JSON.parse(event.data);

          // Update debug info for overlay
          setDebugInfo(result);

          // Only fire callback when a word is fully accepted
          if (result.status === "accepted" && result.word) {
            onWordDetected(result.word, result.confidence);
          }
        } catch (e) {
          console.error("Bad server response:", e);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        setWsConnected(false);
      };

      wsRef.current = ws;
    }

    connect();
    return () => wsRef.current?.close();
  }, []);

  // ── MediaPipe Holistic ──────────────────────────────────────────
  useEffect(() => {
    if (!isActive) {
      cancelAnimationFrame(rafRef.current);
      holisticRef.current?.close();
      holisticRef.current = null;
      const canvas = canvasRef.current;
      if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      setHandsDetected(false);
      setDebugInfo(null);
      return;
    }

    const video = videoRef.current;
    if (!video) return;

    let stopped = false;

    function waitForVideo(cb) {
      if (video.readyState >= 2 && video.videoWidth > 0) cb();
      else video.addEventListener("loadeddata", cb, { once: true });
    }

    waitForVideo(() => {
      if (stopped) return;

      const Holistic = window.Holistic;
      if (!Holistic) {
        console.error("MediaPipe Holistic not loaded");
        return;
      }

      const holistic = new Holistic({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
      });

      holistic.setOptions({
        modelComplexity:        1,
        smoothLandmarks:        true,
        enableSegmentation:     false,
        refineFaceLandmarks:    false,
        minDetectionConfidence: 0.6,
        minTrackingConfidence:  0.6,
      });

      holistic.onResults((results) => {
        const canvas = canvasRef.current;
        const video  = videoRef.current;
        if (!canvas || !video || stopped) return;

        const ctx = canvas.getContext("2d");
        const w   = video.videoWidth  || video.clientWidth;
        const h   = video.videoHeight || video.clientHeight;

        if (canvas.width !== w)  canvas.width  = w;
        if (canvas.height !== h) canvas.height = h;
        ctx.clearRect(0, 0, w, h);

        const { drawConnectors, drawLandmarks, HAND_CONNECTIONS, POSE_CONNECTIONS } = window;

        const hasLeft  = results.leftHandLandmarks?.length  > 0;
        const hasRight = results.rightHandLandmarks?.length > 0;
        setHandsDetected(hasLeft || hasRight);

        // Draw pose skeleton
        if (results.poseLandmarks && POSE_CONNECTIONS) {
          drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
            color: "rgba(255,255,255,0.15)", lineWidth: 1,
          });
        }

        // Draw left hand (teal)
        if (hasLeft) {
          drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {
            color: "#4ecda4", lineWidth: 2,
          });
          drawLandmarks(ctx, results.leftHandLandmarks, {
            color: "#ffffff", fillColor: "#4c3bbd", lineWidth: 1, radius: 3,
          });
        }

        // Draw right hand (blue)
        if (hasRight) {
          drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {
            color: "#5a8ef0", lineWidth: 2,
          });
          drawLandmarks(ctx, results.rightHandLandmarks, {
            color: "#ffffff", fillColor: "#c9608a", lineWidth: 1, radius: 3,
          });
        }

        // Send to server every frame, including empty payloads when no hand is visible
        const ws = wsRef.current;
        if (ws?.readyState === WebSocket.OPEN) {
          // Get conversation context from parent if available
          let context = "";
          if (getContextFunc && typeof getContextFunc === "function") {
            context = getContextFunc();
          }
          
          ws.send(JSON.stringify({
            pose:       results.poseLandmarks      || [],
            left_hand:  results.leftHandLandmarks  || [],
            right_hand: results.rightHandLandmarks || [],
            context:    context, // Send conversation context for filtering
          }));
        }
      });

      holisticRef.current = holistic;

      let processing = false;
      async function loop() {
        if (stopped) return;
        if (!processing && video.readyState >= 2 && video.videoWidth > 0) {
          processing = true;
          try { await holisticRef.current.send({ image: video }); }
          catch (e) { /* ignore */ }
          processing = false;
        }
        rafRef.current = requestAnimationFrame(loop);
      }

      setTimeout(() => { if (!stopped) loop(); }, 800);
    });

    return () => {
      stopped = true;
      cancelAnimationFrame(rafRef.current);
      holisticRef.current?.close();
      holisticRef.current = null;
      setHandsDetected(false);
      const canvas = canvasRef.current;
      if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [isActive]);

  return { canvasRef, handsDetected, wsConnected, debugInfo };
}