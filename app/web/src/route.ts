import { useEffect, useState } from "react";
import type { IconName } from "./components/Icon";

export type Tab = "monitor" | "classify" | "results" | "about";

export interface Page {
  key: Tab;
  slug: string;
  label: string;
  short: string;
  icon: IconName;
}

/** Pages in navigation order. The slug is the hash route, for example #/replay. */
export const PAGES: Page[] = [
  { key: "monitor", slug: "replay", label: "Traffic Replay", short: "Replay", icon: "activity" },
  { key: "classify", slug: "classification", label: "Classification", short: "Classify", icon: "crosshair" },
  { key: "results", slug: "evaluation", label: "Model Evaluation", short: "Evaluation", icon: "chart" },
  { key: "about", slug: "dataset", label: "Dataset and Method", short: "Dataset", icon: "book" },
];

const fromHash = (hash: string): Tab => PAGES.find((p) => p.slug === hash.replace(/^#\/?/, ""))?.key ?? "monitor";

/** The current page, kept in the URL hash so pages can be linked and the back button works. */
export function useRoute(): Tab {
  const [tab, setTab] = useState<Tab>(() => fromHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setTab(fromHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return tab;
}
