import { createContext, useCallback, useState } from "react";

export type Theme = "light" | "dark";

export const ThemeContext = createContext<Theme>("dark");

// index.html sets data-theme before the first paint (saved choice, otherwise dark)
const resolved = (): Theme => (document.documentElement.dataset.theme === "light" ? "light" : "dark");

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(resolved);
  const toggle = useCallback(() => {
    setTheme((current) => {
      const next: Theme = current === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        localStorage.setItem("theme", next);
      } catch {
        /* storage can be unavailable (private mode); the choice then lasts for this visit only */
      }
      return next;
    });
  }, []);
  return [theme, toggle];
}

/*
 * Heatmap ramps. Cell text keeps one color per theme, so each ramp stays inside the range where that
 * color holds at least 4.5:1 contrast (checked across the whole ramp): dark theme cells stay dark enough
 * for light text, light theme cells stay light enough for dark text.
 *   accent  correct or high values
 *   warn    misclassifications
 */
export type RampName = "accent" | "warn";

const RAMPS: Record<Theme, Record<RampName, [string, string]>> = {
  dark: { accent: ["#101720", "#177a72"], warn: ["#101720", "#8c6211"] },
  light: { accent: ["#e8eff3", "#4fb0a7"], warn: ["#ece8dc", "#dca233"] },
};
const TEXT: Record<Theme, string> = { dark: "#eef3f7", light: "#0f1923" };

const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

export function heatColor(value: number, theme: Theme, ramp: RampName = "accent"): { background: string; color: string } {
  const [from, to] = RAMPS[theme][ramp].map(rgb);
  const t = Math.min(1, Math.max(0, value));
  const [r, g, b] = from.map((c, k) => Math.round(c + (to[k] - c) * t));
  return { background: `rgb(${r}, ${g}, ${b})`, color: TEXT[theme] };
}
