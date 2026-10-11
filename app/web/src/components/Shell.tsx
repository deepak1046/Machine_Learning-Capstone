import type { Connection } from "../backend";
import { modelName } from "../format";
import { PAGES, type Tab } from "../route";
import type { Theme } from "../theme";
import type { ModelInfo } from "../types";
import { Icon } from "./Icon";
import { Led, type Tone } from "./ui";

/** Traffic from the left meets a gate: most flows are stopped, one passes through. */
export function Mark({ size = 28 }: { size?: number }) {
  return (
    <svg className="mark" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <rect className="mark__bg" width="32" height="32" rx="4" />
      <path className="mark__flow" d="M6 10h10M6 22h10" />
      <path className="mark__flow mark__flow--pass" d="M6 16h20" />
      <path className="mark__gate" d="M20 6v20" />
    </svg>
  );
}

function Brand() {
  return (
    <div className="brand">
      <Mark />
      <div className="brand__text">
        <span className="brand__name">DDoS Detection</span>
        <span className="brand__sub">CICDDoS2019 flow classifier</span>
      </div>
    </div>
  );
}

function NavLinks({ tab, variant }: { tab: Tab; variant: "side" | "bottom" }) {
  return (
    <ul className={`nav nav--${variant}`}>
      {PAGES.map((page) => (
        <li key={page.key}>
          <a href={`#/${page.slug}`} className="nav__link" aria-current={tab === page.key ? "page" : undefined}>
            <Icon name={page.icon} size={variant === "side" ? 18 : 20} />
            <span>{variant === "side" ? page.label : page.short}</span>
          </a>
        </li>
      ))}
    </ul>
  );
}

const BACKEND: Record<Connection, { tone: Tone; text: string }> = {
  online: { tone: "ok", text: "Connected" },
  offline: { tone: "threat", text: "Offline" },
  connecting: { tone: "warn", text: "Connecting" },
};

function SystemStatus({ connection, models }: { connection: Connection; models: ModelInfo[] | null }) {
  const backend = BACKEND[connection];
  return (
    <section className="system" aria-label="System status">
      <h2 className="overline">System</h2>
      <ul className="system__list">
        <li>
          <Led tone={backend.tone} />
          <span className="system__name">Backend</span>
          <span className="system__state">{backend.text}</span>
        </li>
        {models?.map((m) => {
          // after a lost connection the flags are the last known ones, not current
          const known = connection === "online";
          const tone: Tone = !known ? "neutral" : m.loaded ? "ok" : "warn";
          return (
            <li key={m.key}>
              <Led tone={tone} />
              <span className="system__name" title={modelName(m)}>
                {modelName(m)}
              </span>
              <span className="system__state">{!known ? "Unknown" : m.loaded ? "Loaded" : "Loading"}</span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function ThemeButton({ theme, onToggle, withLabel }: { theme: Theme; onToggle: () => void; withLabel: boolean }) {
  const target = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className={`btn btn--quiet${withLabel ? "" : " btn--icon"}`}
      onClick={onToggle}
      aria-label={`Switch to ${target} theme`}
      title={`Switch to ${target} theme`}
    >
      <Icon name={theme === "dark" ? "sun" : "moon"} size={16} />
      {withLabel && <span>{target === "light" ? "Light theme" : "Dark theme"}</span>}
    </button>
  );
}

interface Props {
  tab: Tab;
  connection: Connection;
  models: ModelInfo[] | null;
  theme: Theme;
  onToggleTheme: () => void;
}

export function Sidebar({ tab, connection, models, theme, onToggleTheme }: Props) {
  return (
    <aside className="sidebar">
      <Brand />
      <nav aria-label="Pages">
        <NavLinks tab={tab} variant="side" />
      </nav>
      <div className="sidebar__foot">
        <SystemStatus connection={connection} models={models} />
        <ThemeButton theme={theme} onToggle={onToggleTheme} withLabel />
      </div>
    </aside>
  );
}

/** Compact header for tablet and phone widths, where the sidebar is replaced by a bottom tab bar. */
export function Topbar({ connection, theme, onToggleTheme }: Omit<Props, "tab" | "models">) {
  const backend = BACKEND[connection];
  return (
    <header className="topbar">
      <Brand />
      <div className="topbar__end">
        <span className="topbar__status">
          <Led tone={backend.tone} />
          {backend.text}
        </span>
        <ThemeButton theme={theme} onToggle={onToggleTheme} withLabel={false} />
      </div>
    </header>
  );
}

export function BottomNav({ tab }: { tab: Tab }) {
  return (
    <nav className="bottom-nav" aria-label="Pages">
      <NavLinks tab={tab} variant="bottom" />
    </nav>
  );
}
