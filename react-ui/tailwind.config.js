/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'security-bg': '#111318',
        'security-text': '#e8e8e8',
        'security-border': '#2a2f3a',
        'security-primary': '#375dfb',
        'security-success': '#44ff44',
        'security-card': '#171a21',
      }
    },
  },
  plugins: [],
}
