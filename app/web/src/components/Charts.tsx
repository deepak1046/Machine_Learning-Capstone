import { Fragment, useContext, type CSSProperties } from "react";
import { signed } from "../format";
import { ThemeContext, heatColor } from "../theme";

export interface Series {
  name: string;
  color: string;
}

export interface BarRow {
  label: string;
  values: (number | null)[];
}

interface BarGroupsProps {
  rows: BarRow[];
  series: Series[];
  max?: number;
  ceiling?: number;
  ceilingLabel?: string;
  labelRows?: string[];
  format?: (value: number) => string;
  showAxis?: boolean;
  labelWidth?: number;
  ariaLabel: string;
}

/** Horizontal bars, one group per row, optionally with a vertical reference line (the ceiling). */
export function BarGroups({
  rows,
  series,
  max = 1,
  ceiling,
  ceilingLabel = "ceiling",
  labelRows = [],
  format = (v) => v.toFixed(3),
  showAxis = true,
  labelWidth,
  ariaLabel,
}: BarGroupsProps) {
  const at = (v: number) => `calc(var(--label-w) + (100% - var(--label-w)) * ${v / max})`;
  const ticks = showAxis ? [0, 0.2, 0.4, 0.6, 0.8, 1].filter((t) => t <= max + 1e-9) : [];

  // a value label that would run into the reference line moves to the far side of the line
  const labelStyle = (v: number): CSSProperties | undefined =>
    ceiling != null && v > 0 && v < ceiling && ceiling - v < 0.075 * max
      ? { left: `${100 + (100 * (ceiling - v)) / v}%` }
      : undefined;

  return (
    <div
      className="bars"
      role="img"
      aria-label={ariaLabel}
      style={labelWidth ? ({ "--label-w": `${labelWidth}px` } as CSSProperties) : undefined}
    >
      {(series.length > 1 || ceiling != null) && (
        <div className="bars__legend">
          {series.length > 1 &&
            series.map((s) => (
              <span key={s.name}>
                <i style={{ background: s.color }} />
                {s.name}
              </span>
            ))}
          {ceiling != null && (
            <span>
              <i className="bars__swatch-line" />
              {ceilingLabel} {format(ceiling)}
            </span>
          )}
        </div>
      )}
      <div className="bars__body">
        {ticks.map((t) => (
          <div className="bars__grid" key={t} style={{ left: at(t) }} />
        ))}
        {rows.map((row) => (
          <div className="bars__row" key={row.label}>
            <div className="bars__label" title={row.label}>
              {row.label}
            </div>
            <div className="bars__track">
              {row.values.map((v, i) =>
                v == null ? (
                  <div className="bars__missing" key={i}>
                    {series.length > 1 ? `${series[i].name}: not run` : "not run"}
                  </div>
                ) : (
                  <div
                    className="bars__bar"
                    key={i}
                    style={{ width: `${(v / max) * 100}%`, background: series[i].color }}
                    title={`${row.label}, ${series[i].name}: ${format(v)}`}
                  >
                    {labelRows.includes(row.label) && <span style={labelStyle(v)}>{format(v)}</span>}
                  </div>
                ),
              )}
            </div>
          </div>
        ))}
        {ceiling != null && <div className="bars__ceiling" style={{ left: at(ceiling) }} />}
      </div>
      {showAxis && (
        <div className="bars__axis" aria-hidden="true">
          {ticks.map((t) => (
            <span key={t} style={{ left: at(t) }}>
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

interface HeatmapProps {
  rows: string[];
  cols: { top: string; bottom?: string }[];
  values: (number | null)[][];
  cellText?: (value: number, row: number, col: number) => string;
  cellTitle?: (value: number | null, row: number, col: number) => string;
  /** Overrides the default accent ramp, for example to color misclassifications differently. */
  cellStyle?: (value: number, row: number, col: number) => CSSProperties;
  gapAfterCol?: number;
  verticalHeaders?: boolean;
  corner?: string;
  maxWidth?: number;
  ariaLabel: string;
}

/** A grid of cells shaded by value; the printed numbers make it its own table view. */
export function Heatmap({
  rows,
  cols,
  values,
  cellText = (v) => v.toFixed(2),
  cellTitle,
  cellStyle,
  gapAfterCol,
  verticalHeaders = false,
  corner,
  maxWidth,
  ariaLabel,
}: HeatmapProps) {
  const theme = useContext(ThemeContext);
  const after = (c: number) => (gapAfterCol != null && c === gapAfterCol + 1 ? " heat__gap" : "");

  return (
    <div className="heat-wrap" role="region" aria-label={ariaLabel} tabIndex={0}>
      <div
        className="heat"
        role="table"
        aria-label={ariaLabel}
        style={{
          gridTemplateColumns: `max-content repeat(${cols.length}, minmax(${verticalHeaders ? 28 : 62}px, 1fr))`,
          maxWidth,
        }}
      >
        <div className="heat__corner">{corner}</div>
        {cols.map((col, c) => (
          <div key={c} className={`heat__colhead${verticalHeaders ? " heat__colhead--vertical" : ""}${after(c)}`}>
            <span>{col.top}</span>
            {col.bottom && <small>{col.bottom}</small>}
          </div>
        ))}
        {rows.map((row, r) => (
          <Fragment key={row}>
            <div className="heat__rowhead">{row}</div>
            {values[r].map((v, c) =>
              v == null ? (
                <div className={`heat__cell heat__cell--none${after(c)}`} key={c} title={cellTitle?.(null, r, c)} />
              ) : (
                <div
                  className={`heat__cell${after(c)}`}
                  key={c}
                  style={cellStyle ? cellStyle(v, r, c) : heatColor(v, theme)}
                  title={cellTitle?.(v, r, c)}
                >
                  {cellText(v, r, c)}
                </div>
              ),
            )}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

export interface ContribItem {
  label: string;
  detail: string;
  value: number;
}

/** Diverging bars around a centre line: right pushes toward the predicted class, left pushes away. */
export function ContribBars({ items, target }: { items: ContribItem[]; target: string }) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return (
    <div className="contrib" role="img" aria-label={`Feature contributions toward ${target}`}>
      <div className="legend">
        <span>
          <i className="legend__swatch legend__swatch--pos" />
          Pushes toward {target}
        </span>
        <span>
          <i className="legend__swatch legend__swatch--neg" />
          Pushes away from {target}
        </span>
      </div>
      {items.map((item) => {
        const width = (Math.abs(item.value) / max) * 42; // leaves room for the value label inside the track
        const side = item.value >= 0 ? "left" : "right";
        return (
          <div className="contrib__row" key={item.label}>
            <div className="contrib__label">
              <span>{item.label}</span>
              {item.detail && <small>{item.detail}</small>}
            </div>
            <div className="contrib__track">
              <div className="contrib__center" />
              <div
                className={`contrib__bar ${item.value >= 0 ? "contrib__bar--pos" : "contrib__bar--neg"}`}
                style={{ width: `${width}%`, [side]: "50%" }}
                title={`${item.label}: ${signed(item.value)}`}
              />
              <span className="contrib__num" style={{ [side]: `calc(50% + ${width}% + 6px)` }}>
                {signed(item.value)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
