"use client";

import { motion } from "framer-motion";

/** Navbar — Floating glassmorphism navigation bar. */
export default function Navbar() {
  const links = [
    { label: "Features", href: "#features" },
    { label: "Demo", href: "#demo" },
    { label: "Analytics", href: "#analytics" },
    { label: "Pricing", href: "#pricing" },
  ];

  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, delay: 0.2 }}
      className="fixed top-0 left-0 right-0 z-50 flex justify-center pt-4 px-4"
    >
      <div className="glass-strong rounded-2xl px-6 py-3 flex items-center gap-8 max-w-4xl w-full">
        {/* Logo */}
        <a href="#" className="flex items-center gap-2 shrink-0">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-glow-cyan to-glow-purple flex items-center justify-center">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#050816" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
          </div>
          <span className="font-bold text-lg text-white hidden sm:block">SpamShield</span>
        </a>

        {/* Links */}
        <div className="hidden md:flex items-center gap-6 flex-1 justify-center">
          {links.map((l) => (
            <a
              key={l.label}
              href={l.href}
              className="text-sm text-text-secondary hover:text-glow-cyan transition-colors duration-300"
            >
              {l.label}
            </a>
          ))}
        </div>

        {/* CTA */}
        <a
          href="#demo"
          className="ml-auto px-4 py-2 text-sm font-semibold rounded-xl bg-gradient-to-r from-glow-cyan to-glow-purple text-bg-primary hover:shadow-[0_0_25px_rgba(0,245,255,0.4)] transition-shadow duration-300"
        >
          Try Demo
        </a>
      </div>
    </motion.nav>
  );
}
