import { useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { clock, count } from "../format";
import type { Severity } from "./ui";

export interface TrafficPoint {
  id: number;
  time: number;
  total: number;
  attack: number;
  /** Highest alert severity raised in this second, if any. */
  severity: Severity | null;
}

interface Props {
  points: TrafficPoint[];
  /** Number of one-second slots across the plot; newer data enters from the right. */
  slots: number;
  height?: number;
}

const PAD = { top: 8, right: 10, left: 42 };
const LANE_H = 6;
const LANE_GAP = 8;
const AXIS_H = 20;
const AXIS_LABEL_GAP = 96; // px, least distance between two timestamps

function niceScale(peak: number) {
  const target = Math.max(peak * 1.08, 4);
  const rough = target / 4;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ?? rough;
  const max = Math.ceil(target / step) * step;
  const ticks: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += step) ticks.push(Math.round(v * 100) / 100);
  return { max, ticks };
}

/** Flows per second as lines: all flows, and the share the model flagged as attack. */
export function LineChart({ points, slots, height = 260 }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const summaryId = useId();

  // measured before the first paint so the chart never flashes at a wrong width
  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const fit = (w: number) => setWidth(Math.max(280, Math.floor(w)));
    fit(el.clientWidth);
    const observer = new ResizeObserver(([entry]) => fit(entry.contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const visible = points.slice(-slots);
  const offset = slots - visible.length;
  const peak = visible.reduce((m, p) => Math.max(m, p.total), 0);
  const { max, ticks } = useMemo(() => niceScale(peak), [peak]);

  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - LANE_GAP - LANE_H - AXIS_H;
  const laneY = PAD.top + plotH + LANE_GAP;
  const step = plotW / Math.max(1, slots - 1);
  const x = (slot: number) => PAD.left + slot * step;
  const y = (v: number) => PAD.top + plotH - (v / max) * plotH;

  const line = (pick: (p: TrafficPoint) => number) =>
    visible.map((p, i) => `${x(offset + i).toFixed(1)},${y(pick(p)).toFixed(1)}`).join(" ");
  const attackLine = line((p) => p.attack);
  // a flat red line along zero would read as an alert, so the flagged series only appears once something was flagged
  const anyFlagged = visible.some((p) => p.attack > 0);
  const area = visible.length
    ? `${x(offset)},${y(0)} ${attackLine} ${x(offset + visible.length - 1)},${y(0)}`
    : "";

  const active = hover ?? cursor;
  const point = active != null ? visible[active] : undefined;
  const last = visible[visible.length - 1];

  const slotAt = (event: PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const slot = Math.round(((event.clientX - box.left) / box.width) * (slots - 1));
    const index = slot - offset;
    return index >= 0 && index < visible.length ? index : null;
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!visible.length) return;
    const from = cursor ?? visible.length - 1;
    let next: number | null = null;
    if (event.key === "ArrowLeft") next = Math.max(0, from - 1);
    else if (event.key === "ArrowRight") next = Math.min(visible.length - 1, from + 1);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = visible.length - 1;
    else if (event.key === "Escape") setCursor(null);
    if (next == null) return;
    event.preventDefault();
    setCursor(next);
  };

  const tipLeft = point ? x(offset + (active ?? 0)) : 0;
  const tipOnLeft = tipLeft > width * 0.58;
  // Timestamps sit under the samples they name. The newest is always shown, then the middle and the oldest
  // when there is room, so a partly filled window is not labeled as if it spanned the whole axis.
  const axisLabels = (() => {
    if (!visible.length) return [];
    const newest = visible.length - 1;
    const placed: { index: number; px: number }[] = [];
    for (const index of [newest, Math.floor(newest / 2), 0]) {
      const px = x(offset + index);
      if (placed.every((q) => Math.abs(q.px - px) >= AXIS_LABEL_GAP)) placed.push({ index, px });
    }
    return placed.map(({ index, px }) => ({
      point: visible[index],
      px,
      anchor: (px < PAD.left + AXIS_LABEL_GAP / 2 ? "start" : px > PAD.left + plotW - AXIS_LABEL_GAP / 2 ? "end" : "middle") as
        | "start"
        | "middle"
        | "end",
    }));
  })();

  return (
    <div className="lc">
      <div className="lc__legend">
        <span className="lc__key">
          <i className="lc__swatch lc__swatch--total" />
          All flows
        </span>
        <span className="lc__key">
          <i className="lc__swatch lc__swatch--attack" />
          Flagged as attack
        </span>
        <span className="lc__key">
          <i className="lc__swatch lc__swatch--medium" />
          Medium alert
        </span>
        <span className="lc__key">
          <i className="lc__swatch lc__swatch--high" />
          High or critical alert
        </span>
        <span className="lc__unit">flows per second</span>
      </div>

      <div
        ref={wrap}
        className="lc__plot"
        role="group"
        aria-label="Traffic volume chart. Use the left and right arrow keys to inspect each second."
        aria-describedby={summaryId}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onFocus={() => visible.length && setCursor((c) => c ?? visible.length - 1)}
        onBlur={() => setCursor(null)}
      >
        <p className="sr-only" id={summaryId}>
          {last
            ? `Latest second: ${count(last.total)} flows, ${count(last.attack)} flagged as attack. Peak in this window: ${count(peak)} flows per second.`
            : "No data yet."}
        </p>
        {width > 0 && (
        <svg width={width} height={height} aria-hidden="true" focusable="false">
          {ticks.map((t) => (
            <g key={t}>
              <line className="lc__grid" x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} />
              <text className="lc__tick" x={PAD.left - 8} y={y(t)} textAnchor="end" dominantBaseline="middle">
                {count(t)}
              </text>
            </g>
          ))}

          {visible.length > 0 && (
            <>
              {anyFlagged && <polygon className="lc__area" points={area} />}
              <polyline className="lc__line lc__line--total" points={line((p) => p.total)} />
              {anyFlagged && <polyline className="lc__line lc__line--attack" points={attackLine} />}
              <circle className="lc__dot lc__dot--total" cx={x(offset + visible.length - 1)} cy={y(last.total)} r={2.5} />
              {visible.map((p, i) =>
                p.severity ? (
                  <rect
                    key={p.id}
                    className={`lc__lane lc__lane--${p.severity}`}
                    x={x(offset + i) - Math.max(1.5, step - 1) / 2}
                    y={laneY}
                    width={Math.max(1.5, step - 1)}
                    height={LANE_H}
                  />
                ) : null,
              )}
            </>
          )}

          {axisLabels.map(({ point: p, px, anchor }) => (
            <text key={p.id} className="lc__tick" x={px} y={height - 4} textAnchor={anchor}>
              {clock(p.time)}
            </text>
          ))}

          {visible.length === 0 && (
            <text className="lc__empty" x={PAD.left + plotW / 2} y={PAD.top + plotH / 2} textAnchor="middle">
              Waiting for the first batch
            </text>
          )}

          {point && active != null && (
            <g>
              <line className="lc__cursor" x1={x(offset + active)} x2={x(offset + active)} y1={PAD.top} y2={laneY + LANE_H} />
              <circle className="lc__dot lc__dot--total" cx={x(offset + active)} cy={y(point.total)} r={3.5} />
              {anyFlagged && <circle className="lc__dot lc__dot--attack" cx={x(offset + active)} cy={y(point.attack)} r={3.5} />}
            </g>
          )}

          <rect
            x={PAD.left}
            y={PAD.top}
            width={plotW}
            height={plotH + LANE_GAP + LANE_H}
            fill="transparent"
            onPointerMove={(event) => setHover(slotAt(event))}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        )}

        {point && active != null && (
          <div
            className="lc__tip"
            style={{ left: tipOnLeft ? undefined : tipLeft + 12, right: tipOnLeft ? width - tipLeft + 12 : undefined }}
          >
            <div className="lc__tip-time">{clock(point.time)}</div>
            <dl>
              <div>
                <dt>All flows</dt>
                <dd>{count(point.total)}</dd>
              </div>
              <div>
                <dt>Flagged</dt>
                <dd>{count(point.attack)}</dd>
              </div>
              <div>
                <dt>Benign</dt>
                <dd>{count(point.total - point.attack)}</dd>
              </div>
              {point.severity && (
                <div>
                  <dt>Top alert</dt>
                  <dd>{point.severity[0].toUpperCase() + point.severity.slice(1)}</dd>
                </div>
              )}
            </dl>
          </div>
        )}
        <span className="sr-only" aria-live="polite">
          {cursor != null && visible[cursor]
            ? `${clock(visible[cursor].time)}: ${count(visible[cursor].total)} flows, ${count(visible[cursor].attack)} flagged.`
            : ""}
        </span>
      </div>
    </div>
  );
}
