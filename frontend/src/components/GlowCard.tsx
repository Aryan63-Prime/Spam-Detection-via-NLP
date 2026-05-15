"use client";

import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useRef, type ReactNode } from "react";

/**
 * GlowCard — Glassmorphism card with mouse-tracking tilt + glow.
 * Follows UI.md: hover tilt effects, glow interactions, animated borders.
 */
export default function GlowCard({
  children,
  className = "",
  glowColor = "cyan",
}: {
  children: ReactNode;
  className?: string;
  glowColor?: "cyan" | "purple" | "green" | "red";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const mouseX = useMotionValue(0.5);
  const mouseY = useMotionValue(0.5);

  const rotateX = useSpring(useTransform(mouseY, [0, 1], [8, -8]), {
    stiffness: 200,
    damping: 30,
  });
  const rotateY = useSpring(useTransform(mouseX, [0, 1], [-8, 8]), {
    stiffness: 200,
    damping: 30,
  });

  const glowColors = {
    cyan: "0, 245, 255",
    purple: "123, 97, 255",
    green: "0, 255, 163",
    red: "255, 61, 113",
  };
  const rgb = glowColors[glowColor];

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    mouseX.set((e.clientX - rect.left) / rect.width);
    mouseY.set((e.clientY - rect.top) / rect.height);
  };

  const handleMouseLeave = () => {
    mouseX.set(0.5);
    mouseY.set(0.5);
  };

  return (
    <motion.div
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ rotateX, rotateY, transformPerspective: 800 }}
      whileHover={{ scale: 1.02 }}
      className={`
        relative rounded-2xl overflow-hidden cursor-pointer
        bg-white/[0.03] backdrop-blur-xl
        border border-white/[0.06]
        transition-shadow duration-500
        hover:shadow-[0_0_40px_rgba(${rgb},0.15)]
        ${className}
      `}
    >
      {/* Glow gradient that follows mouse */}
      <motion.div
        className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-500 pointer-events-none"
        style={{
          background: `radial-gradient(600px circle at ${mouseX.get() * 100}% ${mouseY.get() * 100}%, rgba(${rgb}, 0.06), transparent 40%)`,
        }}
      />
      <div className="relative z-10">{children}</div>
    </motion.div>
  );
}
