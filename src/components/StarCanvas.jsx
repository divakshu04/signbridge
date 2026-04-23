import { useEffect, useRef } from "react";

export default function StarCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animId;
    let stars = [];
    let t = 0;

    function init() {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      stars = Array.from({ length: 160 }, () => ({
        x:     Math.random() * canvas.width,
        y:     Math.random() * canvas.height,
        r:     Math.random() * 1.4 + 0.3,
        o:     Math.random() * 0.7 + 0.2,
        speed: Math.random() * 0.006 + 0.002,
        phase: Math.random() * Math.PI * 2,
      }));
    }

    function draw() {
      const { width: W, height: H } = canvas;
      ctx.clearRect(0, 0, W, H);

      // gradient bg
      const bg = ctx.createLinearGradient(0, 0, 0, H);
      bg.addColorStop(0,   "#0a0820");
      bg.addColorStop(0.4, "#130f35");
      bg.addColorStop(0.8, "#1a0d3a");
      bg.addColorStop(1,   "#0a0820");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      t += 0.01;
      stars.forEach((s) => {
        const op = s.o * (0.6 + 0.4 * Math.sin(t * s.speed * 60 + s.phase));
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200,190,255,${op})`;
        ctx.fill();
      });

      // random sparkle
      if (Math.random() < 0.04) {
        const sx = Math.random() * W;
        const sy = Math.random() * H * 0.8;
        const sg = ctx.createRadialGradient(sx, sy, 0, sx, sy, 6);
        sg.addColorStop(0, "rgba(255,255,255,0.8)");
        sg.addColorStop(1, "rgba(255,255,255,0)");
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fillStyle = sg;
        ctx.fill();
      }

      animId = requestAnimationFrame(draw);
    }

    init();
    draw();

    const onResize = () => init();
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  );
}