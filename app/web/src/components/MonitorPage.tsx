import { useEffect, useMemo, useRef, useState } from "react";
import { streamTick } from "../api";
import { NA, clock, count, duration, modelName, pct, protocolName, si } from "../format";
import { FALLBACK_ADVICE, PLAYBOOK } from "../playbook";
import type { ModelInfo, StreamFlow, StreamTick } from "../types";
import { BarGroups } from "./Charts";
import { Icon } from "./Icon";
import { LineChart, type TrafficPoint } from "./LineChart";
import {
  EmptyState,
  Field,
  Led,
  Notice,
  PageHeader,
  Pager,
  Panel,
  Readouts,
  Segmented,
  Select,
  SeverityTag,
  SortTh,
  Tag,
  TableScroll,
  type Severity,
  type Tone,
} from "./ui";

const TICK_MS = 1000;
const HISTORY = 300; // seconds of traffic kept
const BASE_FLOWS = 30; // flows per second of baseline traffic; a flood multiplies it
const LOG_KEPT = 300;
const PAGE_SIZE = 10;
const STATUS_WINDOW = 5; // seconds the status looks back over
const SUSPICIOUS_FROM = 0.15;
const ATTACK_FROM = 0.5;

type Level = "normal" | "elevated" | "attack";
type LogTab = "alerts" | "flows";
type SortKey = "time" | "severity" | "flows";

interface Tick extends TrafficPoint {
  vectors: Record<string, number>;
}

interface Alert {
  id: string;
  time: number;
  vector: string;
  flows: number;
  confidence: number;
  peakRate: number;
  severity: Severity;
}

interface Seen {
  id: number;
  time: number;
  flow: StreamFlow;
}

interface Totals {
  flows: number;
  flagged: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  ms: number;
  ticks: number;
}

interface Data {
  ticks: Tick[];
  alerts: Alert[];
  recent: Seen[];
  totals: Totals;
  attackSince: number | null;
  counter: number;
}

const EMPTY: Data = {
  ticks: [],
  alerts: [],
  recent: [],
  totals: { flows: 0, flagged: 0, tp: 0, fp: 0, fn: 0, tn: 0, ms: 0, ticks: 0 },
  attackSince: null,
  counter: 0,
};

const RANK: Record<Severity, number> = { low: 0, medium: 1, high: 2, critical: 3 };

/** Severity counts how many flows of one attack type were flagged within a second. */
const severityOf = (flows: number): Severity =>
  flows >= 20 ? "critical" : flows >= 10 ? "high" : flows >= 4 ? "medium" : "low";

const WINDOWS = [
  { value: "60", label: "1 min", seconds: 60 },
  { value: "180", label: "3 min", seconds: 180 },
  { value: "300", label: "5 min", seconds: 300 },
] as const;

const MIN_SEVERITY = [
  { value: "all", label: "All severities", min: 0 },
  { value: "medium", label: "Medium and above", min: 1 },
  { value: "high", label: "High and above", min: 2 },
] as const;

function recentShare(ticks: Tick[]): number {
  const last = ticks.slice(-STATUS_WINDOW);
  const total = last.reduce((sum, t) => sum + t.total, 0);
  return total ? last.reduce((sum, t) => sum + t.attack, 0) / total : 0;
}

const levelOf = (share: number): Level =>
  share >= ATTACK_FROM ? "attack" : share >= SUSPICIOUS_FROM ? "elevated" : "normal";

function tally(ticks: Tick[]): [string, number][] {
  const sums: Record<string, number> = {};
  for (const t of ticks) for (const [v, n] of Object.entries(t.vectors)) sums[v] = (sums[v] ?? 0) + n;
  return Object.entries(sums).sort((a, b) => b[1] - a[1]);
}

function ingest(prev: Data, res: StreamTick, now: number): Data {
  const id = prev.counter + 1;
  const per: Record<string, { n: number; conf: number; rate: number }> = {};
  const totals = { ...prev.totals };
  let attack = 0;

  for (const f of res.flows) {
    const flagged = f.prediction !== "BENIGN";
    if (flagged) {
      attack++;
      const v = (per[f.prediction] ??= { n: 0, conf: 0, rate: 0 });
      v.n++;
      v.conf += f.confidence;
      v.rate = Math.max(v.rate, f.packets_per_s ?? 0);
    }
    if (f.truth != null) {
      const real = f.truth !== "BENIGN";
      if (flagged) real ? totals.tp++ : totals.fp++;
      else real ? totals.fn++ : totals.tn++;
    }
  }
  totals.flows += res.n_flows;
  totals.flagged += attack;
  totals.ms += res.elapsed_ms;
  totals.ticks += 1;

  const fresh: Alert[] = Object.entries(per)
    .sort((a, b) => b[1].n - a[1].n)
    .map(([vector, s]) => ({
      id: `${id}-${vector}`,
      time: now,
      vector,
      flows: s.n,
      confidence: s.conf / s.n,
      peakRate: s.rate,
      severity: severityOf(s.n),
    }));

  const tick: Tick = {
    id,
    time: now,
    total: res.n_flows,
    attack,
    severity: fresh.reduce<Severity | null>((top, a) => (top == null || RANK[a.severity] > RANK[top] ? a.severity : top), null),
    vectors: Object.fromEntries(Object.entries(per).map(([v, s]) => [v, s.n])),
  };
  const ticks = [...prev.ticks, tick].slice(-HISTORY);
  const seen = res.flows.map((flow, i) => ({ id: id * 1000 + i, time: now, flow }));
  const underAttack = levelOf(recentShare(ticks)) === "attack";

  return {
    ticks,
    alerts: [...fresh, ...prev.alerts].slice(0, LOG_KEPT),
    recent: [...seen.reverse(), ...prev.recent].slice(0, LOG_KEPT),
    totals,
    attackSince: underAttack ? (prev.attackSince ?? now) : null,
    counter: id,
  };
}

const STATE: Record<Level, { label: string; tone: Tone }> = {
  normal: { label: "Normal", tone: "ok" },
  elevated: { label: "Suspicious", tone: "warn" },
  attack: { label: "Attack flagged", tone: "threat" },
};

interface Props {
  models: ModelInfo[];
  active: boolean;
}

export function MonitorPage({ models, active }: Props) {
  const [modelKey, setModelKey] = useState("xgb13");
  const [playing, setPlaying] = useState(true);
  const [attackOn, setAttackOn] = useState(false);
  const [share, setShare] = useState(0.7);
  const [vector, setVector] = useState("");
  const [windowKey, setWindowKey] = useState<(typeof WINDOWS)[number]["value"]>("60");
  const [data, setData] = useState<Data>(EMPTY);
  const [error, setError] = useState<string | null>(null);

  const [logTab, setLogTab] = useState<LogTab>("alerts");
  const [minSeverity, setMinSeverity] = useState<(typeof MIN_SEVERITY)[number]["value"]>("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "time", dir: "desc" });
  const [page, setPage] = useState(0);

  const model = models.find((m) => m.key === modelKey) ?? models[0];
  const windowSeconds = WINDOWS.find((w) => w.value === windowKey)?.seconds ?? 60;
  const windowLabel = WINDOWS.find((w) => w.value === windowKey)?.label ?? "1 min";
  const attackTypes = useMemo(
    () => (models.find((m) => m.key === "xgb13")?.classes ?? []).filter((c) => c !== "BENIGN"),
    [models],
  );

  const settings = useRef({ model: modelKey, attackOn, share, vector, ramp: 0, seed: Math.floor(Math.random() * 1e9) });
  useEffect(() => {
    Object.assign(settings.current, { model: modelKey, attackOn, share, vector });
  }, [modelKey, attackOn, share, vector]);

  useEffect(() => {
    if (!active || !playing) return;
    let cancelled = false;
    let timer = 0;

    const step = async () => {
      const s = settings.current;
      const target = s.attackOn ? s.share : 0;
      s.ramp += (target - s.ramp) * 0.4; // a flood builds over a few seconds instead of switching on
      if (Math.abs(target - s.ramp) < 0.03) s.ramp = target;
      const n = Math.min(300, Math.round(BASE_FLOWS * (1 + 3 * s.ramp) * (0.85 + Math.random() * 0.3)));

      let delay = TICK_MS;
      try {
        const res = await streamTick({
          model: s.model,
          n,
          seed: s.seed++,
          attack_share: s.ramp,
          vectors: s.vector ? [s.vector] : null,
        });
        if (!cancelled && res.model === settings.current.model) {
          setData((prev) => ingest(prev, res, Date.now()));
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
        delay = 3000;
      }
      if (!cancelled) timer = window.setTimeout(step, delay);
    };
    void step();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [active, playing, modelKey]);

  const reset = () => {
    settings.current.ramp = 0;
    setAttackOn(false);
    setData(EMPTY);
    setPage(0);
  };

  const switchModel = (key: string) => {
    setModelKey(key);
    setData(EMPTY);
    setPage(0);
  };

  const flagged = recentShare(data.ticks);
  const level = levelOf(flagged);
  const lastFive = data.ticks.slice(-STATUS_WINDOW);
  const lead = tally(lastFive)[0];
  const leadShare = lead ? lead[1] / tally(lastFive).reduce((sum, [, n]) => sum + n, 0) : 0;
  const rate = lastFive.length ? lastFive.reduce((sum, t) => sum + t.total, 0) / lastFive.length : 0;
  const last = data.ticks[data.ticks.length - 1];
  const t = data.totals;
  // while results stop arriving, the last verdict must not be presented as current
  const interrupted = error != null && last != null;
  const state = interrupted
    ? { label: "Replay interrupted", tone: "warn" as Tone }
    : last
      ? STATE[level]
      : { label: "Starting", tone: "neutral" as Tone };

  const inWindow = useMemo(() => data.ticks.slice(-windowSeconds), [data.ticks, windowSeconds]);
  const vectorRows = useMemo(() => tally(inWindow).map(([label, n]) => ({ label, values: [n] })), [inWindow]);
  const guidance = useMemo(
    () =>
      tally(data.ticks.slice(-10))
        .filter(([, n]) => n >= 3)
        .slice(0, 3),
    [data.ticks],
  );

  // protocol mix and typical rates of the kept flows, split by the model's verdict
  const profile = useMemo(() => {
    const median = (values: (number | null)[]) => {
      const v = values.filter((x): x is number => x != null).sort((a, b) => a - b);
      if (!v.length) return null;
      const mid = Math.floor(v.length / 2);
      return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
    };
    const summarize = (flows: StreamFlow[]) => ({
      flows: flows.length,
      udp: flows.length ? flows.filter((f) => f.protocol === 17).length / flows.length : null,
      tcp: flows.length ? flows.filter((f) => f.protocol === 6).length / flows.length : null,
      packets: median(flows.map((f) => f.packets_per_s)),
      bytes: median(flows.map((f) => f.bytes_per_s)),
    });
    return [
      { label: "Passed as benign", ...summarize(data.recent.filter((r) => r.flow.prediction === "BENIGN").map((r) => r.flow)) },
      { label: "Flagged as attack", ...summarize(data.recent.filter((r) => r.flow.prediction !== "BENIGN").map((r) => r.flow)) },
    ];
  }, [data.recent]);

  const detection = t.tp + t.fn > 0 ? t.tp / (t.tp + t.fn) : null;
  const falsePositive = t.fp + t.tn > 0 ? t.fp / (t.fp + t.tn) : null;

  const detail = interrupted
    ? `Results stopped arriving. The last status was ${STATE[level].label.toLowerCase()}, with ${pct(flagged, 0)} of flows flagged.`
    : !last
    ? `Waiting for the first batch from ${model ? modelName(model) : "the model"}.`
    : `${pct(flagged, 0)} of flows flagged in the last ${STATUS_WINDOW} s${
        level !== "normal" && lead ? `, mostly ${lead[0]} (${pct(leadShare, 0)} of flagged flows)` : ""
      }${level === "attack" && data.attackSince ? `. Sustained for ${duration(last.time - data.attackSince)}` : ""}.`;

  const alertTypes = useMemo(() => [...new Set(data.alerts.map((a) => a.vector))].sort(), [data.alerts]);
  const filteredAlerts = useMemo(() => {
    const min = MIN_SEVERITY.find((s) => s.value === minSeverity)?.min ?? 0;
    const q = query.trim().toLowerCase();
    const rows = data.alerts.filter(
      (a) =>
        RANK[a.severity] >= min &&
        (typeFilter === "all" || a.vector === typeFilter) &&
        (!q || a.vector.toLowerCase().includes(q) || clock(a.time).includes(q)),
    );
    const sign = sort.dir === "asc" ? 1 : -1;
    const by = {
      time: (a: Alert, b: Alert) => a.time - b.time,
      severity: (a: Alert, b: Alert) => RANK[a.severity] - RANK[b.severity] || a.time - b.time,
      flows: (a: Alert, b: Alert) => a.flows - b.flows || a.time - b.time,
    }[sort.key];
    return [...rows].sort((a, b) => sign * by(a, b));
  }, [data.alerts, minSeverity, typeFilter, query, sort]);
  const filteredFlows = useMemo(
    () => (onlyFlagged ? data.recent.filter((r) => r.flow.prediction !== "BENIGN") : data.recent),
    [data.recent, onlyFlagged],
  );

  const total = logTab === "alerts" ? filteredAlerts.length : filteredFlows.length;
  const current = Math.min(page, Math.max(0, Math.ceil(total / PAGE_SIZE) - 1));
  const slice = <T,>(rows: T[]) => rows.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);
  const filtering = minSeverity !== "all" || typeFilter !== "all" || query.trim() !== "";

  const sortBy = (key: SortKey) => {
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" }));
    setPage(0);
  };
  const sortProps = (key: SortKey) => ({ active: sort.key === key, dir: sort.dir, onSort: () => sortBy(key) });

  return (
    <div className="page">
      <PageHeader
        title="Traffic Replay"
        description="Held-out CICDDoS2019 flows are replayed once per second and scored by the selected model. This is a simulation, not captured network traffic."
        actions={
          <>
            <button type="button" className="btn" onClick={() => setPlaying((p) => !p)}>
              <Icon name={playing ? "pause" : "play"} />
              {playing ? "Pause" : "Resume"}
            </button>
            <button type="button" className="btn" onClick={reset} title="Clear the history and the counters">
              <Icon name="refresh" />
              Reset
            </button>
          </>
        }
      />

      {error && (
        <Notice tone="threat" title="Replay interrupted">
          {error}
        </Notice>
      )}

      <div className="toolbar" role="group" aria-label="Replay settings">
        <Select label="Detection model" value={modelKey} onChange={(e) => switchModel(e.target.value)}>
          {models.map((m) => (
            <option key={m.key} value={m.key}>
              {modelName(m)}
            </option>
          ))}
        </Select>
        <Select label="Attack type to inject" value={vector} onChange={(e) => setVector(e.target.value)}>
          <option value="">All attack types</option>
          {attackTypes.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </Select>
        <Field label="Attack share of flows" value={pct(share, 0)}>
          <input
            type="range"
            min={0.2}
            max={0.95}
            step={0.05}
            value={share}
            onChange={(e) => setShare(Number(e.target.value))}
          />
        </Field>
        <button
          type="button"
          className={`btn ${attackOn ? "btn--threat-solid" : "btn--threat"}`}
          aria-pressed={attackOn}
          onClick={() => setAttackOn((a) => !a)}
        >
          <Icon name={attackOn ? "close" : "zap"} />
          {attackOn ? "Stop injection" : "Inject attack traffic"}
        </button>
        {model && model.heldout.macro_f1 != null && (
          <p className="toolbar__score">
            <span className="overline">Held-out score of this model</span>
            <span className="mono">
              macro-F1 {model.heldout.macro_f1.toFixed(3)}
              {model.heldout.ceiling_macro_f1 != null && `, ceiling ${model.heldout.ceiling_macro_f1.toFixed(3)}`}
            </span>
          </p>
        )}
      </div>

      <section className={`status status--${state.tone}`} aria-label="Detection status">
        <div className="status__main">
          <div className="status__label">
            <Led tone={state.tone} />
            <span className="overline">Detection status</span>
            <Tag tone="neutral" title="The flows come from a labeled dataset, not from a network">
              Replay
            </Tag>
            {!playing && <Tag tone="warn">Paused</Tag>}
          </div>
          <p className="status__state">{state.label}</p>
          <p className="status__detail">{detail}</p>
        </div>
        <Readouts
          label="Replay measurements"
          items={[
            { label: "Traffic", value: rate ? `${Math.round(rate)}/s` : NA, sub: `flows, last ${STATUS_WINDOW} s` },
            {
              label: "Flagged",
              value: last ? pct(flagged, 0) : NA,
              sub: `last ${STATUS_WINDOW} s`,
              tone: level === "attack" ? "threat" : level === "elevated" ? "warn" : "neutral",
            },
            { label: "Flows analyzed", value: count(t.flows), sub: last ? `${count(last.total)} in the last second` : "since start" },
            {
              label: "Detection rate",
              value: detection == null ? NA : pct(detection),
              sub: detection == null ? "no attack flows yet" : `${count(t.tp)} of ${count(t.tp + t.fn)} attack flows`,
              hint: "Share of attack flows that the model flagged, measured against the dataset labels",
            },
            {
              label: "False positive rate",
              value: falsePositive == null ? NA : pct(falsePositive),
              sub: falsePositive == null ? "no benign flows yet" : `${count(t.fp)} of ${count(t.fp + t.tn)} benign flows`,
              hint: "Share of benign flows that the model flagged as attack, measured against the dataset labels",
            },
            {
              label: "Inference time",
              value: t.ticks ? `${Math.round(t.ms / t.ticks)} ms` : NA,
              sub: t.ticks ? `per batch of about ${Math.round(t.flows / t.ticks)} flows` : "per batch",
            },
          ]}
        />
        <p className="status__rule">
          Status rule: the flagged share of the last {STATUS_WINDOW} seconds, suspicious from {pct(SUSPICIOUS_FROM, 0)}, attack
          from {pct(ATTACK_FROM, 0)}. Flagged means the model assigned an attack class. It is a prediction, not a confirmed
          incident.
        </p>
      </section>

      <div className="layout layout--replay">
        <Panel
          title="Traffic volume"
          meta={
            <Segmented
              label="Time range"
              value={windowKey}
              onChange={setWindowKey}
              options={WINDOWS.map((w) => ({ value: w.value, label: w.label }))}
            />
          }
        >
          <LineChart points={data.ticks} slots={windowSeconds} />
        </Panel>

        <Panel title="Flagged attack types" meta={<span className="faint">last {windowLabel}</span>}>
          {vectorRows.length === 0 ? (
            <p className="hint">No flows flagged in this time range.</p>
          ) : (
            <BarGroups
              rows={vectorRows}
              series={[{ name: "flagged flows", color: "var(--c-threat-fill)" }]}
              max={Math.max(...vectorRows.map((r) => r.values[0])) * 1.18}
              showAxis={false}
              labelWidth={104}
              format={count}
              labelRows={vectorRows.map((r) => r.label)}
              ariaLabel={`Flows flagged per attack type over the last ${windowLabel}`}
            />
          )}
        </Panel>
      </div>

      <Panel title="Response guidance" meta={<span className="faint">general advice, based on the last 10 s</span>}>
        {guidance.length === 0 ? (
          <p className="hint">
            Guidance for an attack type appears here once the model flags it. The app never changes anything on a network.
          </p>
        ) : (
          <ul className="guidance">
            {guidance.map(([v, n]) => (
              <li key={v}>
                <div className="guidance__head">
                  <strong>{v}</strong>
                  <span className="faint mono">{n} flows</span>
                </div>
                <p>{PLAYBOOK[v] ?? FALLBACK_ADVICE}</p>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title="Traffic profile" meta={<span className="faint">last {data.recent.length} flows</span>} flush>
        <TableScroll label="Protocol mix and typical rates, benign against flagged flows">
          <table className="dt">
            <thead>
              <tr>
                <th scope="col">Verdict</th>
                <th scope="col" className="num">
                  Flows
                </th>
                <th scope="col" className="num">
                  UDP
                </th>
                <th scope="col" className="num">
                  TCP
                </th>
                <th scope="col" className="num">
                  Median packets/s
                </th>
                <th scope="col" className="num">
                  Median bytes/s
                </th>
              </tr>
            </thead>
            <tbody>
              {profile.map((row) => (
                <tr key={row.label}>
                  <td>{row.label}</td>
                  <td className="num mono">{count(row.flows)}</td>
                  <td className="num mono">{row.udp == null ? NA : pct(row.udp, 0)}</td>
                  <td className="num mono">{row.tcp == null ? NA : pct(row.tcp, 0)}</td>
                  <td className="num mono">{si(row.packets)}</td>
                  <td className="num mono">{si(row.bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableScroll>
      </Panel>

      <Panel
        title="Detection log"
        flush
        meta={
          <Segmented
            variant="tab"
            label="Log view"
            value={logTab}
            onChange={(v) => {
              setLogTab(v);
              setPage(0);
            }}
            options={[
              { value: "alerts", label: "Alerts", count: data.alerts.length, controls: "log-panel" },
              { value: "flows", label: "Flows", count: data.recent.length, controls: "log-panel" },
            ]}
          />
        }
      >
        <div id="log-panel" role="tabpanel" aria-label={logTab === "alerts" ? "Alerts" : "Flows"}>
          {logTab === "alerts" ? (
            <div className="filters">
              <div className="search">
                <Icon name="search" size={14} />
                <input
                  type="search"
                  aria-label="Search alerts by attack type or time"
                  placeholder="Search attack type or time"
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    setPage(0);
                  }}
                />
              </div>
              <Select
                label="Severity"
                value={minSeverity}
                onChange={(e) => {
                  setMinSeverity(e.target.value as typeof minSeverity);
                  setPage(0);
                }}
              >
                {MIN_SEVERITY.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </Select>
              <Select
                label="Attack type"
                value={typeFilter}
                onChange={(e) => {
                  setTypeFilter(e.target.value);
                  setPage(0);
                }}
              >
                <option value="all">All types</option>
                {alertTypes.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </Select>
            </div>
          ) : (
            <div className="filters">
              <label className="check">
                <input
                  type="checkbox"
                  checked={onlyFlagged}
                  onChange={(e) => {
                    setOnlyFlagged(e.target.checked);
                    setPage(0);
                  }}
                />
                Flagged flows only
              </label>
            </div>
          )}

          <TableScroll label={logTab === "alerts" ? "Alerts table" : "Flows table"}>
            {logTab === "alerts" ? (
              <table className="dt">
                <thead>
                  <tr>
                    <SortTh label="Time" {...sortProps("time")} />
                    <SortTh label="Severity" {...sortProps("severity")} />
                    <th scope="col">Attack type</th>
                    <SortTh label="Flows" numeric {...sortProps("flows")} />
                    <th scope="col" className="num col-wide">
                      Peak packets/s
                    </th>
                    <th scope="col" className="num col-wide">
                      Mean confidence
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {slice(filteredAlerts).map((a) => (
                    <tr key={a.id}>
                      <td className="mono">{clock(a.time)}</td>
                      <td>
                        <SeverityTag severity={a.severity} />
                      </td>
                      <td>{a.vector}</td>
                      <td className="num mono">{a.flows}</td>
                      <td className="num mono col-wide">{si(a.peakRate)}</td>
                      <td className="num mono col-wide">{pct(a.confidence, 0)}</td>
                    </tr>
                  ))}
                  {filteredAlerts.length === 0 && (
                    <tr>
                      <td colSpan={6}>
                        <p className="dt__empty">
                          {filtering
                            ? "No alerts match these filters."
                            : "No alerts yet. Inject attack traffic to see how the model reacts."}
                        </p>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            ) : (
              <table className="dt">
                <thead>
                  <tr>
                    <th scope="col">Time</th>
                    <th scope="col">Prediction</th>
                    <th scope="col">Protocol</th>
                    <th scope="col" className="num">
                      Packets/s
                    </th>
                    <th scope="col" className="num col-wide">
                      Bytes/s
                    </th>
                    <th scope="col" className="col-wide">
                      Confidence
                    </th>
                    <th scope="col">Dataset label</th>
                  </tr>
                </thead>
                <tbody>
                  {slice(filteredFlows).map(({ id, time, flow }) => {
                    const attack = flow.prediction !== "BENIGN";
                    return (
                      <tr key={id}>
                        <td className="mono">{clock(time)}</td>
                        <td>
                          {attack ? (
                            <Tag tone="threat" icon="alert">
                              {flow.prediction}
                            </Tag>
                          ) : (
                            <span className="faint">Benign</span>
                          )}
                        </td>
                        <td className="mono">{protocolName(flow.protocol)}</td>
                        <td className="num mono">{si(flow.packets_per_s)}</td>
                        <td className="num mono col-wide">{si(flow.bytes_per_s)}</td>
                        <td className="col-wide">
                          <span className="conf mono">
                            {pct(flow.confidence, 0)}
                            <span className="conf__track">
                              <i style={{ width: `${flow.confidence * 100}%` }} />
                            </span>
                          </span>
                        </td>
                        <td>
                          {flow.truth == null ? (
                            <span className="faint">{NA}</span>
                          ) : flow.truth === flow.prediction ? (
                            <span className="muted">{flow.truth}</span>
                          ) : (
                            <Tag tone="warn" title="The model disagrees with the dataset label">
                              {flow.truth}
                            </Tag>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                  {filteredFlows.length === 0 && (
                    <tr>
                      <td colSpan={7}>
                        <p className="dt__empty">
                          {data.recent.length ? "No flagged flows in the kept history." : "Waiting for traffic."}
                        </p>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            )}
          </TableScroll>

          <div className="log-foot">
            <Pager page={current} pageSize={PAGE_SIZE} total={total} onPage={setPage} />
            <p className="hint">
              {logTab === "alerts"
                ? "Severity counts the flows of one attack type flagged within a second: medium from 4, high from 10, critical from 20. It is a count heuristic, not a risk score."
                : "The dataset label is the ground truth stored with each replayed flow. Live traffic would not carry one. Amber marks a prediction that differs from it."}
            </p>
          </div>
        </div>
      </Panel>

      {!last && !error && playing && active && (
        <p className="sr-only" role="status">
          Starting the replay.
        </p>
      )}
      {!playing && data.ticks.length === 0 && (
        <EmptyState icon="pause" title="The replay is paused">
          Resume it to score the first batch of flows.
        </EmptyState>
      )}
    </div>
  );
}
