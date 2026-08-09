/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Each color resolves through a CSS custom property (defined in
        // index.css) rather than a static hex value. This is what makes
        // dark mode work correctly everywhere at once: every component
        // just uses `bg-surface-raised` / `text-ink` etc., and the actual
        // RGB values flip when the `light` class toggles on <html> - no
        // component needs its own light/dark variant classes.
        surface: {
          DEFAULT: "rgb(var(--color-surface) / <alpha-value>)",
          raised: "rgb(var(--color-surface-raised) / <alpha-value>)",
          sunken: "rgb(var(--color-surface-sunken) / <alpha-value>)",
        },
        ink: {
          DEFAULT: "rgb(var(--color-ink) / <alpha-value>)",
          muted: "rgb(var(--color-ink-muted) / <alpha-value>)",
        },
        mustard: {
          DEFAULT: "rgb(var(--color-mustard) / <alpha-value>)",
          soft: "rgb(var(--color-mustard-soft) / <alpha-value>)",
        },
        brick: {
          DEFAULT: "rgb(var(--color-brick) / <alpha-value>)",
          soft: "rgb(var(--color-brick-soft) / <alpha-value>)",
        },
        sage: {
          DEFAULT: "rgb(var(--color-sage) / <alpha-value>)",
          soft: "rgb(var(--color-sage-soft) / <alpha-value>)",
        },
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        body: ["Inter", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        card: "6px",
      },
    },
  },
  plugins: [],
};
