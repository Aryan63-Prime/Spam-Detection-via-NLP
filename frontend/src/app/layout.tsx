import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "SpamShield AI — AI-Powered Communication Defense",
  description:
    "Industry-grade AI spam detection platform. Detect spam, phishing, scams, and malicious messages with transformer-powered NLP and explainable AI.",
  keywords: [
    "spam detection",
    "AI security",
    "phishing defense",
    "NLP",
    "machine learning",
    "cybersecurity",
  ],
  openGraph: {
    title: "SpamShield AI — AI-Powered Communication Defense",
    description:
      "Detect spam, phishing, scams, and malicious messages before they reach users.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} dark`}>
      <body className="bg-bg-primary text-text-primary antialiased">
        {children}
      </body>
    </html>
  );
}
