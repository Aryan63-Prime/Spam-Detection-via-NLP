"use client";

import { motion } from "framer-motion";
import AnimatedButton from "./AnimatedButton";

/** Hero Section — Cinematic AI defense hero with animated orb + copy. */
export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-4 overflow-hidden">
      {/* Gradient Background Blobs */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] rounded-full bg-glow-cyan/[0.04] blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] rounded-full bg-glow-purple/[0.06] blur-[100px]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] rounded-full bg-glow-green/[0.03] blur-[80px]" />
      </div>

      {/* Cyber Grid */}
      <div className="absolute inset-0 cyber-grid-bg opacity-40 pointer-events-none" />

      {/* Animated AI Orb */}
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 1.2, ease: "easeOut" }}
        className="relative mb-12"
      >
        <div className="relative w-48 h-48 md:w-64 md:h-64">
          {/* Outer ring */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="absolute inset-0 rounded-full border-2 border-dashed border-glow-cyan/20"
          />
          {/* Middle ring */}
          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
            className="absolute inset-4 rounded-full border border-glow-purple/30"
          />
          {/* Core orb */}
          <motion.div
            animate={{ scale: [1, 1.08, 1] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            className="absolute inset-8 rounded-full bg-gradient-to-br from-glow-cyan/20 via-glow-purple/30 to-glow-green/20 backdrop-blur-sm shadow-[0_0_60px_rgba(0,245,255,0.25),inset_0_0_40px_rgba(123,97,255,0.15)]"
          />
          {/* Inner glow */}
          <div className="absolute inset-12 rounded-full bg-gradient-to-br from-glow-cyan/10 to-glow-purple/10 blur-md" />
          {/* Shield icon center */}
          <div className="absolute inset-0 flex items-center justify-center">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="url(#shieldGrad)" strokeWidth="1.5" strokeLinecap="round">
              <defs>
                <linearGradient id="shieldGrad" x1="0" y1="0" x2="24" y2="24">
                  <stop offset="0%" stopColor="#00F5FF" />
                  <stop offset="100%" stopColor="#7B61FF" />
                </linearGradient>
              </defs>
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          {/* Orbiting dots */}
          {[0, 90, 180, 270].map((deg) => (
            <motion.div
              key={deg}
              animate={{ rotate: 360 }}
              transition={{ duration: 12, repeat: Infinity, ease: "linear", delay: deg / 360 * 12 }}
              className="absolute inset-0"
              style={{ transform: `rotate(${deg}deg)` }}
            >
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-glow-cyan shadow-[0_0_10px_rgba(0,245,255,0.6)]" />
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Status Badge */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.6 }}
        className="glass rounded-full px-4 py-1.5 mb-6 flex items-center gap-2"
      >
        <span className="w-2 h-2 rounded-full bg-glow-green animate-pulse" />
        <span className="text-xs text-text-secondary tracking-wider uppercase">AI Defense System Active</span>
      </motion.div>

      {/* Headline */}
      <motion.h1
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7, duration: 0.8 }}
        className="text-5xl md:text-7xl lg:text-8xl font-bold text-center leading-[1.05] tracking-tight max-w-5xl"
      >
        <span className="gradient-text-cyber">AI-Powered</span>
        <br />
        Communication Defense
      </motion.h1>

      {/* Subheadline */}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1, duration: 0.7 }}
        className="mt-6 text-lg md:text-xl text-text-secondary max-w-2xl text-center leading-relaxed"
      >
        Detect spam, phishing, scams, and malicious messages before they reach users. 
        Powered by transformers, explainable AI, and real-time analytics.
      </motion.p>

      {/* CTAs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.3, duration: 0.6 }}
        className="mt-10 flex flex-wrap gap-4 justify-center"
      >
        <AnimatedButton variant="primary" size="lg">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          Start Scanning
        </AnimatedButton>
        <AnimatedButton variant="secondary" size="lg">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Watch Live Demo
        </AnimatedButton>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="absolute bottom-8 flex flex-col items-center gap-2"
      >
        <span className="text-xs text-text-muted tracking-widest uppercase">Scroll to explore</span>
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="w-5 h-8 rounded-full border border-text-muted/30 flex justify-center pt-1.5"
        >
          <div className="w-1 h-2 rounded-full bg-glow-cyan" />
        </motion.div>
      </motion.div>
    </section>
  );
}
