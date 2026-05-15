"use client";

import { useEffect, useState, useRef } from "react";
import { motion, useInView } from "framer-motion";
import RevealSection from "./RevealSection";

/** Animated counter for stat numbers */
function Counter({ target, suffix = "", duration = 2000 }: { target: number; suffix?: string; duration?: number }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setCount(target); clearInterval(timer); }
      else setCount(Math.floor(start));
    }, 16);
    return () => clearInterval(timer);
  }, [inView, target, duration]);

  return <span ref={ref}>{count.toLocaleString()}{suffix}</span>;
}

const stats = [
  { value: 99.2, suffix: "%", label: "Detection Accuracy", color: "text-glow-green" },
  { value: 42, suffix: "ms", label: "Avg. Response Time", color: "text-glow-cyan" },
  { value: 13, suffix: "", label: "ML Models Trained", color: "text-glow-purple" },
  { value: 2847, suffix: "+", label: "Threats Blocked Today", color: "text-accent-red" },
];

export default function AnalyticsSection() {
  return (
    <section id="analytics" className="relative py-32 px-4">
      <RevealSection className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <span className="text-xs text-glow-green tracking-[0.3em] uppercase font-medium">Analytics</span>
          <h2 className="text-4xl md:text-6xl font-bold mt-3 mb-4">
            Real-Time <span className="gradient-text-cyber">Intelligence</span>
          </h2>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-16">
          {stats.map((s, i) => (
            <RevealSection key={s.label} delay={i * 0.1}>
              <div className="glass rounded-2xl p-6 text-center">
                <p className={`text-3xl md:text-5xl font-bold ${s.color}`}>
                  <Counter target={s.value} suffix={s.suffix} />
                </p>
                <p className="text-sm text-text-muted mt-2">{s.label}</p>
              </div>
            </RevealSection>
          ))}
        </div>

        {/* Dashboard Preview */}
        <RevealSection delay={0.3}>
          <div className="glass-strong rounded-3xl p-6 md:p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-3 h-3 rounded-full bg-accent-red" />
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <div className="w-3 h-3 rounded-full bg-glow-green" />
              <span className="text-xs text-text-muted ml-3">SpamShield Dashboard — Live</span>
            </div>

            {/* Mini Charts */}
            <div className="grid md:grid-cols-3 gap-4">
              {/* Threat Timeline */}
              <div className="bg-white/[0.02] rounded-xl p-4 border border-white/[0.06]">
                <p className="text-xs text-text-muted mb-3">Threat Timeline (7d)</p>
                <div className="flex items-end gap-1 h-24">
                  {[35, 52, 48, 70, 45, 80, 63].map((h, i) => (
                    <motion.div
                      key={i}
                      initial={{ height: 0 }}
                      whileInView={{ height: `${h}%` }}
                      transition={{ delay: i * 0.1, duration: 0.6 }}
                      viewport={{ once: true }}
                      className="flex-1 rounded-t bg-gradient-to-t from-glow-cyan/40 to-glow-cyan/80"
                    />
                  ))}
                </div>
              </div>

              {/* Detection Breakdown */}
              <div className="bg-white/[0.02] rounded-xl p-4 border border-white/[0.06]">
                <p className="text-xs text-text-muted mb-3">Detection Breakdown</p>
                <div className="space-y-3">
                  {[
                    { label: "Phishing", pct: 42, color: "bg-accent-red" },
                    { label: "Spam", pct: 31, color: "bg-glow-purple" },
                    { label: "Scam", pct: 18, color: "bg-accent-blue" },
                    { label: "Safe", pct: 9, color: "bg-glow-green" },
                  ].map((item) => (
                    <div key={item.label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-text-secondary">{item.label}</span>
                        <span className="text-text-muted">{item.pct}%</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-white/[0.05]">
                        <motion.div
                          initial={{ width: 0 }}
                          whileInView={{ width: `${item.pct}%` }}
                          transition={{ duration: 0.8 }}
                          viewport={{ once: true }}
                          className={`h-full rounded-full ${item.color}`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Live Alerts */}
              <div className="bg-white/[0.02] rounded-xl p-4 border border-white/[0.06]">
                <p className="text-xs text-text-muted mb-3">Recent Alerts</p>
                <div className="space-y-2">
                  {[
                    { msg: "Phishing link detected", time: "2s ago", threat: true },
                    { msg: "Bulk spam blocked", time: "15s ago", threat: true },
                    { msg: "Safe email delivered", time: "22s ago", threat: false },
                    { msg: "Scam attempt flagged", time: "41s ago", threat: true },
                  ].map((alert, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className={`w-1.5 h-1.5 rounded-full ${alert.threat ? "bg-accent-red animate-pulse" : "bg-glow-green"}`} />
                      <span className="text-text-secondary flex-1 truncate">{alert.msg}</span>
                      <span className="text-text-muted shrink-0">{alert.time}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </RevealSection>
      </RevealSection>
    </section>
  );
}
