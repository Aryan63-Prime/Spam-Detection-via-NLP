"use client";

import dynamic from "next/dynamic";
import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import Footer from "@/components/Footer";

/* Dynamic imports for heavy sections — keeps initial load fast.
   UI.md rule: "Lazy load 3D scenes", "Use dynamic imports" */
const ParticleField = dynamic(() => import("@/components/ParticleField"), { ssr: false });
const SpamDetectorDemo = dynamic(() => import("@/components/SpamDetectorDemo"), { ssr: false });
const FeaturesSection = dynamic(() => import("@/components/FeaturesSection"));
const AnalyticsSection = dynamic(() => import("@/components/AnalyticsSection"));
const CyberAttackSimulation = dynamic(() => import("@/components/CyberAttackSimulation"), { ssr: false });
const PricingSection = dynamic(() => import("@/components/PricingSection"));
const CTASection = dynamic(() => import("@/components/CTASection"));

/**
 * SpamShield AI — Landing Page
 * ==============================
 * Sections per UI.md:
 * 1. Hero (animated orb + copy + CTAs)
 * 2. Live AI Demo (interactive scanner)
 * 3. Features (glassmorphism cards)
 * 4. Cyber Attack Simulation (viral section)
 * 5. Analytics Dashboard Preview
 * 6. Pricing
 * 7. Final CTA
 * 8. Footer
 */
export default function Home() {
  return (
    <>
      <ParticleField count={40} />
      <Navbar />

      <main>
        <HeroSection />

        {/* Divider glow */}
        <div className="h-px bg-gradient-to-r from-transparent via-glow-cyan/20 to-transparent" />

        <SpamDetectorDemo />

        <div className="h-px bg-gradient-to-r from-transparent via-glow-purple/20 to-transparent" />

        <FeaturesSection />

        <div className="h-px bg-gradient-to-r from-transparent via-accent-red/20 to-transparent" />

        <CyberAttackSimulation />

        <div className="h-px bg-gradient-to-r from-transparent via-glow-green/20 to-transparent" />

        <AnalyticsSection />

        <div className="h-px bg-gradient-to-r from-transparent via-glow-purple/20 to-transparent" />

        <PricingSection />

        <CTASection />
      </main>

      <Footer />
    </>
  );
}
