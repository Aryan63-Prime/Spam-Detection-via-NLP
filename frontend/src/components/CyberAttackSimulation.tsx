"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence, useInView } from "framer-motion";
import RevealSection from "./RevealSection";

const THREATS = [
  { text: "FREE MONEY — CLAIM NOW!", type: "Scam" },
  { text: "Your account has been locked. Verify now.", type: "Phishing" },
  { text: "CLICK HERE to win iPhone 16!", type: "Spam" },
  { text: "URGENT: Wire $500 to unlock funds.", type: "Fraud" },
  { text: "Congratulations! You're the 1,000,000th visitor!", type: "Spam" },
  { text: "Reset your password immediately or lose access.", type: "Phishing" },
];

export default function CyberAttackSimulation() {
  const [active, setActive] = useState<typeof THREATS>([]);
  const [destroyed, setDestroyed] = useState<Set<number>>(new Set());
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-200px" });
  const counterRef = useRef(0);

  useEffect(() => {
    if (!inView) return;

    const interval = setInterval(() => {
      if (counterRef.current >= THREATS.length) {
        clearInterval(interval);
        return;
      }
      setActive((prev) => [...prev, THREATS[counterRef.current]]);
      const idx = counterRef.current;

      // Auto-destroy after 1.5s
      setTimeout(() => {
        setDestroyed((prev) => new Set(prev).add(idx));
      }, 1500);

      counterRef.current++;
    }, 800);

    return () => clearInterval(interval);
  }, [inView]);

  return (
    <section className="relative py-32 px-4 overflow-hidden" ref={ref}>
      <RevealSection className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <span className="text-xs text-accent-red tracking-[0.3em] uppercase font-medium">Live Simulation</span>
          <h2 className="text-4xl md:text-6xl font-bold mt-3 mb-4">
            Cyber Attack <span className="gradient-text-threat">Neutralization</span>
          </h2>
          <p className="text-text-secondary max-w-xl mx-auto">
            Watch as SpamShield AI detects and neutralizes incoming threats in real time.
          </p>
        </div>

        {/* Simulation Arena */}
        <div className="glass-strong rounded-3xl p-6 md:p-10 min-h-[400px] relative overflow-hidden">
          {/* Shield center */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
            <motion.div
              animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
              transition={{ duration: 3, repeat: Infinity }}
              className="w-32 h-32 rounded-full border-2 border-glow-cyan/20 flex items-center justify-center"
            >
              <div className="w-16 h-16 rounded-full bg-glow-cyan/10 flex items-center justify-center">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00F5FF" strokeWidth="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              </div>
            </motion.div>
          </div>

          {/* Threats */}
          <div className="relative z-10 space-y-3">
            <AnimatePresence>
              {active.map((threat, i) => (
                <motion.div
                  key={i}
                  initial={{ x: -200, opacity: 0 }}
                  animate={destroyed.has(i)
                    ? { x: 0, opacity: 0, scale: 0.8, filter: "blur(4px)" }
                    : { x: 0, opacity: 1 }
                  }
                  transition={{ duration: 0.5 }}
                  className={`flex items-center gap-4 p-3 rounded-xl border transition-all duration-500 ${
                    destroyed.has(i)
                      ? "border-glow-green/30 bg-glow-green/[0.03]"
                      : "border-accent-red/30 bg-accent-red/[0.05] animate-threat-pulse"
                  }`}
                >
                  {/* Status */}
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                    destroyed.has(i) ? "bg-glow-green/20" : "bg-accent-red/20"
                  }`}>
                    {destroyed.has(i) ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00FFA3" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                    ) : (
                      <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF3D71" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
                      </motion.div>
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium truncate ${destroyed.has(i) ? "text-text-muted line-through" : "text-white"}`}>
                      {threat.text}
                    </p>
                  </div>

                  {/* Type badge */}
                  <span className={`text-xs px-2 py-0.5 rounded-md font-medium shrink-0 ${
                    destroyed.has(i)
                      ? "bg-glow-green/10 text-glow-green"
                      : "bg-accent-red/10 text-accent-red"
                  }`}>
                    {destroyed.has(i) ? "BLOCKED" : threat.type}
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>

            {active.length === 0 && (
              <div className="text-center py-20 text-text-muted text-sm">
                <motion.p animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 2, repeat: Infinity }}>
                  Waiting for threats...
                </motion.p>
              </div>
            )}
          </div>
        </div>
      </RevealSection>
    </section>
  );
}
