import { useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode, type SelectHTMLAttributes } from "react";
import { Icon, type IconName } from "./Icon";

export type Tone = "neutral" | "accent" | "threat" | "warn" | "ok";
export type Severity = "low" | "medium" | "high" | "critical";

/* ---------- page structure ---------- */

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="page-head">
      <div className="page-head__text">
        <h1 className="page-title">{title}</h1>
        {description && <p className="page-desc">{description}</p>}
      </div>
      {actions && <div className="page-head__actions">{actions}</div>}
    </header>
  );
}

/** A bordered group for content that needs a title and a boundary: a chart, a table, a form. */
export function Panel({
  title,
  meta,
  children,
  flush = false,
  className = "",
  level = 2,
}: {
  title: string;
  meta?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  className?: string;
  /** Heading level of the title: 3 when the panel sits inside a titled Section. */
  level?: 2 | 3;
}) {
  const titleId = useId();
  const Heading = level === 3 ? "h3" : "h2";
  return (
    <section className={`panel ${className}`.trim()} aria-labelledby={titleId}>
      <header className="panel__head">
        <Heading className="panel__title" id={titleId}>
          {title}
        </Heading>
        {meta && <div className="panel__meta">{meta}</div>}
      </header>
      <div className={`panel__body${flush ? " panel__body--flush" : ""}`}>{children}</div>
    </section>
  );
}

/** A titled section without a box, for reading-oriented pages. */
export function Section({ title, children, id }: { title: string; children: ReactNode; id?: string }) {
  const titleId = useId();
  return (
    <section className="section" aria-labelledby={titleId} id={id}>
      <h2 className="section__title" id={titleId}>
        {title}
      </h2>
      {children}
    </section>
  );
}

/* ---------- readings ---------- */

export interface ReadoutItem {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
  hint?: string;
}

const READOUT_MIN_WIDTH = 136; // px, narrowest a readout may get before the strip wraps

/** Columns for n readouts in a strip of the given width: as many as fit, preferring a split with no short last row. */
function readoutColumns(n: number, width: number): number {
  const fit = Math.max(1, Math.floor(width / READOUT_MIN_WIDTH));
  if (n <= fit) return n;
  for (let c = fit; c >= 2; c--) if (n % c === 0) return c;
  return Math.max(1, Math.ceil(n / Math.ceil(n / fit)));
}

/** Instrument-style strip of measurements: one border around all of them, hairlines between. */
export function Readouts({ items, label }: { items: ReadoutItem[]; label: string }) {
  const strip = useRef<HTMLDListElement>(null);
  const [columns, setColumns] = useState(items.length);

  useLayoutEffect(() => {
    const el = strip.current;
    if (!el) return;
    const fit = () => setColumns(readoutColumns(items.length, el.clientWidth));
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(el);
    return () => observer.disconnect();
  }, [items.length]);

  // a last row that is not full hands its leftover cells to the final readout, so the strip never has a hole
  const leftover = columns > 1 && items.length % columns ? columns - (items.length % columns) : 0;

  return (
    <dl
      ref={strip}
      className="readouts"
      aria-label={label}
      style={{ gridTemplateColumns: `repeat(${Math.max(1, columns)}, minmax(0, 1fr))` }}
    >
      {items.map((item, i) => (
        <div
          className={`readout readout--${item.tone ?? "neutral"}`}
          key={item.label}
          title={item.hint}
          style={i === items.length - 1 && leftover ? { gridColumn: `span ${leftover + 1}` } : undefined}
        >
          <dt className="readout__label">{item.label}</dt>
          <dd className="readout__value">{item.value}</dd>
          {item.sub != null && <dd className="readout__sub">{item.sub}</dd>}
        </div>
      ))}
    </dl>
  );
}

/* ---------- status ---------- */

export function Led({ tone = "neutral" }: { tone?: Tone }) {
  return <span className={`led led--${tone}`} aria-hidden="true" />;
}

export function Tag({
  tone = "neutral",
  solid = false,
  icon,
  title,
  children,
}: {
  tone?: Tone;
  solid?: boolean;
  icon?: IconName;
  title?: string;
  children: ReactNode;
}) {
  return (
    <span className={`tag tag--${tone}${solid ? " tag--solid" : ""}`} title={title}>
      {icon && <Icon name={icon} size={12} />}
      {children}
    </span>
  );
}

const SEVERITY_LABEL: Record<Severity, string> = { low: "Low", medium: "Medium", high: "High", critical: "Critical" };

export function SeverityTag({ severity }: { severity: Severity }) {
  const tone: Tone = severity === "low" ? "neutral" : severity === "medium" ? "warn" : "threat";
  return (
    <Tag tone={tone} solid={severity === "critical"} icon={severity === "high" || severity === "critical" ? "alert" : undefined}>
      {SEVERITY_LABEL[severity]}
    </Tag>
  );
}

export function Notice({
  tone = "accent",
  title,
  children,
  action,
}: {
  tone?: Tone;
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const icon: IconName = tone === "threat" || tone === "warn" ? "alert" : tone === "ok" ? "check" : "info";
  return (
    <div className={`notice notice--${tone}`} role={tone === "threat" ? "alert" : "status"}>
      <Icon name={icon} size={16} />
      <div className="notice__body">
        {title && <strong className="notice__title">{title}</strong>}
        {children && <div>{children}</div>}
      </div>
      {action}
    </div>
  );
}

export function EmptyState({
  icon = "info",
  title,
  children,
  busy = false,
}: {
  icon?: IconName;
  title: string;
  children?: ReactNode;
  busy?: boolean;
}) {
  return (
    <div className="empty" role={busy ? "status" : undefined}>
      <span className="empty__icon">{busy ? <span className="spinner" /> : <Icon name={icon} size={20} />}</span>
      <h2 className="empty__title">{title}</h2>
      {children && <div className="empty__text">{children}</div>}
    </div>
  );
}

export const Spinner = () => <span className="spinner" role="presentation" />;

/* ---------- controls ---------- */

interface SegmentedProps<T extends string> {
  label: string;
  options: { value: T; label: ReactNode; count?: number; controls?: string }[];
  value: T;
  onChange: (value: T) => void;
  /** "tab" switches panels, "radio" picks one value. Both support arrow keys, Home and End. */
  variant?: "radio" | "tab";
}

export function Segmented<T extends string>({ label, options, value, onChange, variant = "radio" }: SegmentedProps<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const tab = variant === "tab";

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const last = options.length - 1;
    let next: number | null = null;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") next = index === last ? 0 : index + 1;
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = index === 0 ? last : index - 1;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = last;
    if (next == null) return;
    event.preventDefault();
    onChange(options[next].value);
    refs.current[next]?.focus();
  };

  return (
    <div className="seg" role={tab ? "tablist" : "radiogroup"} aria-label={label}>
      {options.map((option, i) => {
        const on = option.value === value;
        return (
          <button
            key={option.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role={tab ? "tab" : "radio"}
            aria-selected={tab ? on : undefined}
            aria-checked={tab ? undefined : on}
            aria-controls={tab ? option.controls : undefined}
            tabIndex={on ? 0 : -1}
            className={`seg__btn${on ? " is-on" : ""}`}
            onClick={() => onChange(option.value)}
            onKeyDown={(event) => onKeyDown(event, i)}
          >
            {option.label}
            {option.count != null && <span className="seg__count">{option.count}</span>}
          </button>
        );
      })}
    </div>
  );
}

export function Select({
  label,
  children,
  className = "",
  ...rest
}: { label: string } & SelectHTMLAttributes<HTMLSelectElement>) {
  const id = useId();
  return (
    <div className={`field ${className}`.trim()}>
      <label className="field__label" htmlFor={id}>
        {label}
      </label>
      <div className="select">
        <select id={id} {...rest}>
          {children}
        </select>
        <Icon name="chevron-down" size={14} />
      </div>
    </div>
  );
}

/** Label above a control, with an optional live value on the right (used for sliders). */
export function Field({ label, value, children }: { label: string; value?: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field__label field__label--split">
        {label}
        {value != null && <span className="field__value">{value}</span>}
      </span>
      {children}
    </label>
  );
}

/* ---------- tables ---------- */

export function Pager({
  page,
  pageSize,
  total,
  onPage,
}: {
  page: number;
  pageSize: number;
  total: number;
  onPage: (page: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const current = Math.min(page, pages - 1);
  const from = total === 0 ? 0 : current * pageSize + 1;
  const to = Math.min(total, (current + 1) * pageSize);
  return (
    <nav className="pager" aria-label="Pagination">
      <span className="pager__range">
        {from.toLocaleString("en")} to {to.toLocaleString("en")} of {total.toLocaleString("en")}
      </span>
      <div className="pager__buttons">
        <span className="pager__page">
          Page {current + 1} of {pages}
        </span>
        <button
          type="button"
          className="btn btn--icon btn--sm"
          disabled={current === 0}
          onClick={() => onPage(current - 1)}
          aria-label="Previous page"
        >
          <Icon name="chevron-left" size={14} />
        </button>
        <button
          type="button"
          className="btn btn--icon btn--sm"
          disabled={current >= pages - 1}
          onClick={() => onPage(current + 1)}
          aria-label="Next page"
        >
          <Icon name="chevron-right" size={14} />
        </button>
      </div>
    </nav>
  );
}

export function SortTh({
  label,
  active,
  dir,
  onSort,
  numeric = false,
  className = "",
}: {
  label: string;
  active: boolean;
  dir: "asc" | "desc";
  onSort: () => void;
  numeric?: boolean;
  className?: string;
}) {
  return (
    <th
      scope="col"
      className={`${numeric ? "num" : ""} ${className}`.trim() || undefined}
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button type="button" className={`sort${active ? " is-active" : ""}`} onClick={onSort}>
        {label}
        <Icon name={active && dir === "asc" ? "arrow-up" : "arrow-down"} size={12} />
      </button>
    </th>
  );
}

/** Horizontal scroll container that keyboard users can reach. */
export function TableScroll({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="table-scroll" role="region" aria-label={label} tabIndex={0}>
      {children}
    </div>
  );
}
