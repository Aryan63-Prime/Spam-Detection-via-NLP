"use client";

import AnimatedButton from "./AnimatedButton";
import RevealSection from "./RevealSection";

export default function CTASection() {
  return (
    <section className="relative py-32 px-4 overflow-hidden">
      {/* Background glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] rounded-full bg-glow-cyan/[0.06] blur-[150px]" />
      </div>

      <RevealSection className="max-w-3xl mx-auto text-center relative z-10">
        <h2 className="text-4xl md:text-6xl font-bold mb-6">
          Ready to <span className="gradient-text-cyber">Defend</span> Your Communications?
        </h2>
        <p className="text-lg text-text-secondary mb-10 max-w-xl mx-auto">
          Join thousands of organizations using SpamShield AI to protect their users from spam, phishing, and cyber threats.
        </p>
        <div className="flex flex-wrap gap-4 justify-center">
          <AnimatedButton variant="primary" size="lg">
            Get Started Free
          </AnimatedButton>
          <AnimatedButton variant="secondary" size="lg">
            View Documentation
          </AnimatedButton>
        </div>
      </RevealSection>
    </section>
  );
}
