/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        panel: "#0f172a",
        panel2: "#1e293b",
        accent: "#3b82f6",
      },
    },
  },
  plugins: [],
}
