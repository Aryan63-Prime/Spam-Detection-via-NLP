"use client";

import { useEffect, useRef } from "react";

/**
 * ParticleField — Animated particle background.
 * Creates floating data points that drift upward like a neural data stream.
 * GPU-friendly: uses CSS transforms (no JS position updates per frame).
 * Mobile-optimized: reduces particle count automatically.
 */
export default function ParticleField({ count = 50 }: { count?: number }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Reduce particles on mobile
    const isMobile = window.innerWidth < 768;
    const particleCount = isMobile ? Math.floor(count / 3) : count;

    const particles: HTMLDivElement[] = [];

    for (let i = 0; i < particleCount; i++) {
      const p = document.createElement("div");
      const size = Math.random() * 3 + 1;
      const x = Math.random() * 100;
      const delay = Math.random() * 20;
      const duration = Math.random() * 15 + 15;
      const hue = Math.random() > 0.5 ? "0, 245, 255" : "123, 97, 255";
      const opacity = Math.random() * 0.5 + 0.1;

      p.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        left: ${x}%;
        bottom: -10px;
        border-radius: 50%;
        background: rgba(${hue}, ${opacity});
        box-shadow: 0 0 ${size * 4}px rgba(${hue}, ${opacity * 0.5});
        animation: particle-drift ${duration}s linear ${delay}s infinite;
        pointer-events: none;
      `;

      container.appendChild(p);
      particles.push(p);
    }

    return () => {
      particles.forEach((p) => p.remove());
    };
  }, [count]);

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 overflow-hidden pointer-events-none z-0"
      aria-hidden="true"
    />
  );
}
