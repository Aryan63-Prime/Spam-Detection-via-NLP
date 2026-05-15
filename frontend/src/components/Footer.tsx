"use client";

const footerLinks = {
  Product: ["Features", "Pricing", "API Docs", "Changelog"],
  Company: ["About", "Blog", "Careers", "Contact"],
  Resources: ["Documentation", "Tutorials", "GitHub", "Community"],
  Legal: ["Privacy", "Terms", "Security", "Compliance"],
};

export default function Footer() {
  return (
    <footer className="border-t border-white/[0.06] bg-bg-secondary/50">
      <div className="max-w-6xl mx-auto px-4 py-16">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-glow-cyan to-glow-purple flex items-center justify-center">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#050816" strokeWidth="2.5" strokeLinecap="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              </div>
              <span className="font-bold text-white">SpamShield</span>
            </div>
            <p className="text-sm text-text-muted leading-relaxed">
              AI-powered communication defense platform.
            </p>
          </div>

          {/* Links */}
          {Object.entries(footerLinks).map(([title, links]) => (
            <div key={title}>
              <h4 className="text-sm font-semibold text-white mb-4">{title}</h4>
              <ul className="space-y-2">
                {links.map((link) => (
                  <li key={link}>
                    <a href="#" className="text-sm text-text-muted hover:text-glow-cyan transition-colors duration-300">
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom */}
        <div className="mt-12 pt-6 border-t border-white/[0.06] flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-xs text-text-muted">
            © 2026 SpamShield AI. All rights reserved.
          </p>
          <div className="flex items-center gap-4">
            {["GitHub", "Twitter", "LinkedIn"].map((social) => (
              <a key={social} href="#" className="text-xs text-text-muted hover:text-glow-cyan transition-colors">
                {social}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
