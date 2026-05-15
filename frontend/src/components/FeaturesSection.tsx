"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import GlowCard from "./GlowCard";
import RevealSection from "./RevealSection";

const features = [
  {
    icon: "⚡",
    title: "Real-Time Detection",
    desc: "Sub-50ms inference with GPU-accelerated models. Classify messages before they reach the inbox.",
    glow: "cyan" as const,
  },
  {
    icon: "🛡️",
    title: "Phishing Defense",
    desc: "Detect sophisticated phishing attempts, social engineering, and fraudulent links with 99%+ accuracy.",
    glow: "red" as const,
  },
  {
    icon: "🧠",
    title: "Transformer AI",
    desc: "Powered by BERT, DistilBERT, and RoBERTa. Fine-tuned on millions of labeled messages.",
    glow: "purple" as const,
  },
  {
    icon: "🔍",
    title: "Explainable AI",
    desc: "LIME and SHAP explanations show exactly why a message was flagged. Full transparency.",
    glow: "green" as const,
  },
  {
    icon: "🌐",
    title: "Multilingual Support",
    desc: "Process messages in 100+ languages including Hinglish, SMS slang, and Unicode text.",
    glow: "cyan" as const,
  },
  {
    icon: "📊",
    title: "Threat Analytics",
    desc: "Real-time dashboards, heatmaps, and trend analysis. Monitor your threat landscape 24/7.",
    glow: "purple" as const,
  },
];

export default function FeaturesSection() {
  return (
    <section id="features" className="relative py-32 px-4">
      <RevealSection className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <span className="text-xs text-glow-purple tracking-[0.3em] uppercase font-medium">Capabilities</span>
          <h2 className="text-4xl md:text-6xl font-bold mt-3 mb-4">
            Enterprise-Grade <span className="gradient-text-cyber">Protection</span>
          </h2>
          <p className="text-text-secondary max-w-xl mx-auto">
            A comprehensive AI defense system built with 6 traditional ML models, 4 deep learning architectures, and 3 transformer models.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <RevealSection key={f.title} delay={i * 0.1}>
              <GlowCard glowColor={f.glow} className="p-6 h-full">
                <span className="text-3xl">{f.icon}</span>
                <h3 className="text-lg font-bold text-white mt-4 mb-2">{f.title}</h3>
                <p className="text-sm text-text-secondary leading-relaxed">{f.desc}</p>
              </GlowCard>
            </RevealSection>
          ))}
        </div>
      </RevealSection>
    </section>
  );
}
