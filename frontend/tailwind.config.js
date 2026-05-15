/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#050816',
          secondary: '#070B1A',
          tertiary: '#0A0F1F',
        },
        glow: {
          cyan: '#00F5FF',
          purple: '#7B61FF',
          green: '#00FFA3',
        },
        accent: {
          red: '#FF3D71',
          blue: '#5B8CFF',
        },
        text: {
          primary: '#FFFFFF',
          secondary: '#B8C0CC',
          muted: '#6E7A8A',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'glow-pulse': 'glow-pulse 3s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'scan': 'scan 2s linear infinite',
        'threat-pulse': 'threat-pulse 1.5s ease-in-out infinite',
        'border-glow': 'border-glow 3s linear infinite',
        'particle-drift': 'particle-drift 20s linear infinite',
        'shield-rotate': 'shield-rotate 10s linear infinite',
        'typing': 'typing 3.5s steps(40, end)',
      },
      keyframes: {
        'glow-pulse': {
          '0%, 100%': { opacity: '0.4', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.05)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        'scan': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        'threat-pulse': {
          '0%, 100%': { boxShadow: '0 0 20px rgba(255,61,113,0.3)' },
          '50%': { boxShadow: '0 0 40px rgba(255,61,113,0.7)' },
        },
        'border-glow': {
          '0%, 100%': { borderColor: 'rgba(0,245,255,0.3)' },
          '50%': { borderColor: 'rgba(0,245,255,0.8)' },
        },
        'particle-drift': {
          '0%': { transform: 'translateY(100vh) rotate(0deg)' },
          '100%': { transform: 'translateY(-100vh) rotate(720deg)' },
        },
        'shield-rotate': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        'typing': {
          '0%': { width: '0' },
          '100%': { width: '100%' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};
