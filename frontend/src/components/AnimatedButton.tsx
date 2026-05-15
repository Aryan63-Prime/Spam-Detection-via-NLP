"use client";

import { motion } from "framer-motion";
import { type ReactNode } from "react";

/**
 * AnimatedButton — Premium CTA button with glow hover effect.
 * UI.md: "Buttons glow on hover" + neon gradient aesthetics.
 */
export default function AnimatedButton({
  children,
  variant = "primary",
  size = "lg",
  onClick,
  className = "",
}: {
  children: ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  onClick?: () => void;
  className?: string;
}) {
  const sizes = {
    sm: "px-4 py-2 text-sm",
    md: "px-6 py-3 text-base",
    lg: "px-8 py-4 text-lg",
  };

  const variants = {
    primary: `
      bg-gradient-to-r from-glow-cyan via-glow-purple to-glow-green
      text-bg-primary font-bold
      shadow-[0_0_30px_rgba(0,245,255,0.3)]
      hover:shadow-[0_0_50px_rgba(0,245,255,0.5)]
    `,
    secondary: `
      bg-white/[0.05] backdrop-blur-lg
      border border-glow-cyan/30
      text-glow-cyan font-semibold
      hover:bg-white/[0.1]
      hover:border-glow-cyan/60
      hover:shadow-[0_0_30px_rgba(0,245,255,0.2)]
    `,
    ghost: `
      bg-transparent
      text-text-secondary font-medium
      hover:text-glow-cyan
      hover:bg-white/[0.03]
    `,
  };

  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.03, y: -2 }}
      whileTap={{ scale: 0.97 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
      className={`
        relative rounded-xl overflow-hidden
        transition-all duration-300 cursor-pointer
        ${sizes[size]}
        ${variants[variant]}
        ${className}
      `}
    >
      {/* Shine sweep on hover */}
      <div className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-700">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] hover:translate-x-[100%] transition-transform duration-700" />
      </div>
      <span className="relative z-10 flex items-center justify-center gap-2">
        {children}
      </span>
    </motion.button>
  );
}
