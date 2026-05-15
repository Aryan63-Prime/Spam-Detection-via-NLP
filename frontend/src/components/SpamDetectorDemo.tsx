"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import RevealSection from "./RevealSection";

const SAMPLE_TEXTS = [
  "Congratulations! You've won a $1000 gift card. Click here to claim now!",
  "Hey, are we still meeting for lunch tomorrow at noon?",
  "URGENT: Your bank account has been compromised. Verify immediately!",
  "Mom said dinner will be ready at 7. Don't be late!",
  "FREE MONEY!!! Send your details now to claim your prize!!!",
];

interface AnalysisResult {
  prediction: string;
  confidence: number;
  keywords: { word: string; importance: number; isSpam: boolean }[];
  processing_time: number;
}

/** Mock prediction — will connect to real API in production */
function mockPredict(text: string): AnalysisResult {
  const spamWords = ["free", "win", "won", "click", "urgent", "congratulations", "prize", "claim", "money", "bank", "verify", "send", "details", "gift", "card", "!!!"];
  const words = text.toLowerCase().split(/\s+/);
  let spamScore = 0;

  const keywords = words.slice(0, 12).map((w) => {
    const clean = w.replace(/[^a-z]/g, "");
    const isSpam = spamWords.some((s) => clean.includes(s));
    const importance = isSpam ? 0.6 + Math.random() * 0.4 : Math.random() * 0.3;
    if (isSpam) spamScore += importance;
    return { word: w, importance, isSpam };
  });

  const confidence = Math.min(0.99, Math.max(0.05, spamScore / 3));
  const isSpam = confidence > 0.5;

  return {
    prediction: isSpam ? "spam" : "ham",
    confidence: isSpam ? confidence : 1 - confidence,
    keywords,
    processing_time: Math.random() * 30 + 10,
  };
}

export default function SpamDetectorDemo() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyze = useCallback(() => {
    if (!text.trim()) return;
    setIsAnalyzing(true);
    setResult(null);

    setTimeout(() => {
      setResult(mockPredict(text));
      setIsAnalyzing(false);
    }, 1200);
  }, [text]);

  const loadSample = (sample: string) => {
    setText(sample);
    setResult(null);
  };

  const isSpam = result?.prediction === "spam";

  return (
    <section id="demo" className="relative py-32 px-4">
      <RevealSection className="max-w-5xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16">
          <span className="text-xs text-glow-cyan tracking-[0.3em] uppercase font-medium">Live Demo</span>
          <h2 className="text-4xl md:text-6xl font-bold mt-3 mb-4">
            Try the <span className="gradient-text-cyber">AI Scanner</span>
          </h2>
          <p className="text-text-secondary max-w-xl mx-auto">
            Type or paste any message. Our AI analyzes it in real-time, highlighting threats and explaining its reasoning.
          </p>
        </div>

        {/* Demo Card */}
        <div className="glass-strong rounded-3xl p-6 md:p-8 max-w-3xl mx-auto">
          {/* Input */}
          <div className="relative">
            <textarea
              value={text}
              onChange={(e) => { setText(e.target.value); setResult(null); }}
              placeholder="Type a message to scan for threats..."
              rows={3}
              className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl px-5 py-4 text-white placeholder:text-text-muted resize-none focus:outline-none focus:border-glow-cyan/40 focus:shadow-[0_0_20px_rgba(0,245,255,0.1)] transition-all duration-300"
            />
            {/* Scanning overlay */}
            {isAnalyzing && (
              <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none">
                <motion.div
                  animate={{ top: ["0%", "100%", "0%"] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-glow-cyan to-transparent shadow-[0_0_15px_rgba(0,245,255,0.6)]"
                />
              </div>
            )}
          </div>

          {/* Sample Texts */}
          <div className="flex flex-wrap gap-2 mt-4">
            {SAMPLE_TEXTS.map((s, i) => (
              <button
                key={i}
                onClick={() => loadSample(s)}
                className="text-xs px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.06] text-text-muted hover:text-glow-cyan hover:border-glow-cyan/30 transition-all duration-300 truncate max-w-[200px]"
              >
                {s.slice(0, 35)}...
              </button>
            ))}
          </div>

          {/* Analyze Button */}
          <motion.button
            onClick={analyze}
            disabled={!text.trim() || isAnalyzing}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full mt-5 py-3.5 rounded-xl font-semibold text-bg-primary bg-gradient-to-r from-glow-cyan via-glow-purple to-glow-green shadow-[0_0_25px_rgba(0,245,255,0.25)] hover:shadow-[0_0_40px_rgba(0,245,255,0.4)] disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-300"
          >
            {isAnalyzing ? (
              <span className="flex items-center justify-center gap-2">
                <motion.span animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }} className="inline-block w-4 h-4 border-2 border-bg-primary/30 border-t-bg-primary rounded-full" />
                Scanning...
              </span>
            ) : "Analyze Message"}
          </motion.button>

          {/* Results */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20, height: 0 }}
                animate={{ opacity: 1, y: 0, height: "auto" }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.5 }}
                className="mt-6"
              >
                {/* Verdict */}
                <div className={`rounded-xl p-5 border ${isSpam ? "border-accent-red/30 bg-accent-red/[0.05]" : "border-glow-green/30 bg-glow-green/[0.05]"}`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${isSpam ? "bg-accent-red/20" : "bg-glow-green/20"}`}>
                        {isSpam ? (
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FF3D71" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/></svg>
                        ) : (
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00FFA3" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                        )}
                      </div>
                      <div>
                        <p className={`font-bold text-lg ${isSpam ? "text-accent-red" : "text-glow-green"}`}>
                          {isSpam ? "⚠ SPAM DETECTED" : "✓ SAFE MESSAGE"}
                        </p>
                        <p className="text-xs text-text-muted">{result.processing_time.toFixed(1)}ms • DistilBERT</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-2xl font-bold text-white">{(result.confidence * 100).toFixed(1)}%</p>
                      <p className="text-xs text-text-muted">confidence</p>
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <div className="h-2 rounded-full bg-white/[0.05] overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence * 100}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className={`h-full rounded-full ${isSpam ? "bg-gradient-to-r from-accent-red to-accent-red/60" : "bg-gradient-to-r from-glow-green to-glow-green/60"}`}
                    />
                  </div>
                </div>

                {/* Keyword Analysis */}
                <div className="mt-4 rounded-xl p-5 bg-white/[0.02] border border-white/[0.06]">
                  <p className="text-sm font-semibold text-text-secondary mb-3">🔍 Keyword Analysis</p>
                  <div className="flex flex-wrap gap-2">
                    {result.keywords.map((kw, i) => (
                      <motion.span
                        key={i}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.05 }}
                        className={`px-3 py-1 rounded-lg text-sm font-medium border ${
                          kw.isSpam
                            ? "bg-accent-red/10 border-accent-red/30 text-accent-red"
                            : "bg-white/[0.03] border-white/[0.08] text-text-secondary"
                        }`}
                      >
                        {kw.word}
                        <span className="ml-1.5 text-[10px] opacity-60">{(kw.importance * 100).toFixed(0)}%</span>
                      </motion.span>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </RevealSection>
    </section>
  );
}
