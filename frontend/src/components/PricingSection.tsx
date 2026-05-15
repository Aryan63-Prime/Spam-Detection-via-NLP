"use client";

import GlowCard from "./GlowCard";
import AnimatedButton from "./AnimatedButton";
import RevealSection from "./RevealSection";

const plans = [
  {
    name: "Starter",
    price: "Free",
    desc: "For individual developers and hobbyists.",
    features: ["1,000 scans/month", "3 ML models", "API access", "Basic analytics"],
    glow: "cyan" as const,
    cta: "Get Started",
    popular: false,
  },
  {
    name: "Pro",
    price: "$29",
    period: "/mo",
    desc: "For teams and growing businesses.",
    features: ["50,000 scans/month", "All 13 models", "XAI explanations", "Priority support", "Batch API", "Advanced analytics"],
    glow: "purple" as const,
    cta: "Start Free Trial",
    popular: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    desc: "For large organizations with custom needs.",
    features: ["Unlimited scans", "Custom models", "On-premise deploy", "SLA guarantee", "Dedicated support", "SSO & RBAC"],
    glow: "green" as const,
    cta: "Contact Sales",
    popular: false,
  },
];

export default function PricingSection() {
  return (
    <section id="pricing" className="relative py-32 px-4">
      <RevealSection className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <span className="text-xs text-glow-cyan tracking-[0.3em] uppercase font-medium">Pricing</span>
          <h2 className="text-4xl md:text-6xl font-bold mt-3 mb-4">
            Start Protecting <span className="gradient-text-cyber">Today</span>
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {plans.map((plan, i) => (
            <RevealSection key={plan.name} delay={i * 0.15}>
              <GlowCard glowColor={plan.glow} className={`p-6 h-full flex flex-col ${plan.popular ? "ring-1 ring-glow-purple/40" : ""}`}>
                {plan.popular && (
                  <span className="self-start text-[10px] px-2.5 py-0.5 rounded-full bg-glow-purple/20 text-glow-purple font-semibold tracking-wider uppercase mb-3">
                    Most Popular
                  </span>
                )}
                <h3 className="text-xl font-bold text-white">{plan.name}</h3>
                <div className="mt-3 mb-2">
                  <span className="text-4xl font-bold text-white">{plan.price}</span>
                  {plan.period && <span className="text-text-muted text-sm">{plan.period}</span>}
                </div>
                <p className="text-sm text-text-muted mb-6">{plan.desc}</p>
                <ul className="space-y-2 mb-8 flex-1">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-center gap-2 text-sm text-text-secondary">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00FFA3" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                      {f}
                    </li>
                  ))}
                </ul>
                <AnimatedButton variant={plan.popular ? "primary" : "secondary"} size="md" className="w-full">
                  {plan.cta}
                </AnimatedButton>
              </GlowCard>
            </RevealSection>
          ))}
        </div>
      </RevealSection>
    </section>
  );
}
