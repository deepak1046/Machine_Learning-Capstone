import type { ModelInfo } from "./types";

/** Shown wherever a value does not exist (a split that was not run, a flow without a label). */
export const NA = "n/a";

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

export const f3 = (v: number | null | undefined) => (v == null ? NA : v.toFixed(3));

export const count = (v: number) => v.toLocaleString("en");

const compactFormat = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 });

/** A feature value for display: integers in full, large numbers compact, fractions to 4 significant digits. */
export function featureValue(v: number | null | undefined): string {
  if (v == null) return NA;
  if (Math.abs(v) >= 1e5) return compactFormat.format(v);
  if (Number.isInteger(v)) return v.toLocaleString("en");
  return String(Number(v.toPrecision(4)));
}

export const signed = (v: number, digits = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

export const seconds = (ms: number) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

const SI = ["", "K", "M", "G", "T"];

/** A rate with an SI prefix and three significant digits: 1.47e9 -> "1.47 G". */
export function si(v: number | null | undefined): string {
  if (v == null) return NA;
  let x = Math.abs(v);
  let i = 0;
  while ((x >= 1000 || Number(x.toPrecision(3)) >= 1000) && i < SI.length - 1) {
    x /= 1000;
    i++;
  }
  const text = i === 0 ? String(Number(x.toPrecision(3))) : x.toPrecision(3);
  return `${v < 0 ? "−" : ""}${text}${i ? " " + SI[i] : ""}`;
}

export const clock = (ms: number) => new Date(ms).toLocaleTimeString("en-GB");

export function duration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** The two XGBoost models share a title, so the class scheme is part of the name. */
export function modelName(m: Pick<ModelInfo, "title" | "kind" | "level" | "classes">): string {
  if (m.kind !== "tree") return m.title;
  return `${m.title} (${m.classes.length} ${m.level === "attack type" ? "types" : "families"})`;
}

export const protocolName = (p: number | null) =>
  p === 6 ? "TCP" : p === 17 ? "UDP" : p === 1 ? "ICMP" : p == null ? NA : `IP ${p}`;
