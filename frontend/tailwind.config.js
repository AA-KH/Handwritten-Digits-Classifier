/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        void: '#030305',
        surface: '#0d0d14',
        panel: '#12121f',
        border: '#1e1e32',
        accent: '#6c63ff',
        'accent-glow': '#9d97ff',
        neon: '#00d4ff',
        'neon-dim': '#006680',
        emerald: '#00ff9d',
        ember: '#ff6b35',
        muted: '#4a4a6a',
        ghost: '#8888aa',
        text: '#e8e8f0',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'Consolas', 'monospace'],
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['"Inter"', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
        'scan': 'scan 2s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glowPulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
      },
      boxShadow: {
        'glow-accent': '0 0 20px rgba(108, 99, 255, 0.4)',
        'glow-neon': '0 0 20px rgba(0, 212, 255, 0.4)',
        'glow-emerald': '0 0 20px rgba(0, 255, 157, 0.4)',
        'inner-glow': 'inset 0 0 30px rgba(108, 99, 255, 0.1)',
      },
    },
  },
  plugins: [],
}
