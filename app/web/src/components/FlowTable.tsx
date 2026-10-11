import { useMemo, useState } from "react";
import { NA, count, featureValue, pct } from "../format";
import type { Classification, Flow } from "../types";
import { Panel, Pager, Segmented, SortTh, Tag, TableScroll } from "./ui";

type Filter = "all" | "attack" | "benign" | "wrong" | "low";
type SortKey = "id" | "confidence";

const PAGE_SIZE = 12;
const LOW_CONFIDENCE = 0.7;

const SHOWN_FEATURES = [
  { name: "Flow Duration", label: "Duration (µs)", wide: false },
  { name: "Total Fwd Packets", label: "Fwd packets", wide: false },
  { name: "Fwd Packet Length Max", label: "Max fwd packet (B)", wide: true },
  { name: "Source Port", label: "Src port", wide: true },
  { name: "Destination Port", label: "Dst port", wide: true },
];

const FILTERS: { key: Filter; label: string; test: (f: Flow) => boolean; needsTruth?: boolean }[] = [
  { key: "all", label: "All", test: () => true },
  { key: "attack", label: "Attacks", test: (f) => f.prediction !== "BENIGN" },
  { key: "benign", label: "Benign", test: (f) => f.prediction === "BENIGN" },
  { key: "wrong", label: "Misclassified", test: (f) => f.truth != null && f.truth !== f.prediction, needsTruth: true },
  { key: "low", label: "Low confidence", test: (f) => f.confidence < LOW_CONFIDENCE },
];

interface Props {
  result: Classification;
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export function FlowTable({ result, selectedId, onSelect }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "id", dir: "asc" });
  const [page, setPage] = useState(0);

  const hasTruth = result.truth != null;
  const filters = FILTERS.filter((f) => hasTruth || !f.needsTruth);
  const counts = useMemo(
    () => Object.fromEntries(FILTERS.map((f) => [f.key, result.flows.filter(f.test).length])) as Record<Filter, number>,
    [result.flows],
  );
  const columns = useMemo(
    () =>
      SHOWN_FEATURES.map((f) => ({ ...f, index: result.feature_names.indexOf(f.name) })).filter((c) => c.index >= 0),
    [result.feature_names],
  );

  const active = filters.find((f) => f.key === filter) ?? filters[0];
  const rows = useMemo(() => {
    const kept = result.flows.filter(active.test);
    const sign = sort.dir === "asc" ? 1 : -1;
    return [...kept].sort((a, b) => sign * (sort.key === "id" ? a.id - b.id : a.confidence - b.confidence || a.id - b.id));
  }, [result.flows, active, sort]);
  const current = Math.min(page, Math.max(0, Math.ceil(rows.length / PAGE_SIZE) - 1));
  const visible = rows.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);

  const sortProps = (key: SortKey) => ({
    active: sort.key === key,
    dir: sort.dir,
    onSort: () => {
      setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "asc" }));
      setPage(0);
    },
  });

  return (
    <Panel title="Classified flows" flush>
      <div className="filters">
        <Segmented
          label="Filter flows"
          value={active.key}
          onChange={(key) => {
            setFilter(key);
            setPage(0);
          }}
          options={filters.map((f) => ({ value: f.key, label: f.label, count: counts[f.key] }))}
        />
      </div>
      {result.n_flows > result.flows.length && (
        <p className="hint panel__note">
          Showing the first {count(result.flows.length)} of {count(result.n_flows)} scored flows. The charts and metrics
          cover all of them.
        </p>
      )}
      <TableScroll label="Classified flows table">
        <table className="dt dt--select">
          <thead>
            <tr>
              <SortTh label="#" {...sortProps("id")} />
              <th scope="col">Prediction</th>
              <SortTh label="Confidence" {...sortProps("confidence")} />
              {hasTruth && <th scope="col">True label</th>}
              {columns.map((c) => (
                <th key={c.name} scope="col" className={`num${c.wide ? " col-wide" : ""}`} title={c.name}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((f) => {
              const attack = f.prediction !== "BENIGN";
              const wrong = f.truth != null && f.truth !== f.prediction;
              return (
                <tr
                  key={f.id}
                  className={selectedId === f.id ? "is-selected" : undefined}
                  aria-selected={selectedId === f.id}
                  tabIndex={0}
                  onClick={() => onSelect(f.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(f.id);
                    }
                  }}
                >
                  <td className="mono faint">{f.id + 1}</td>
                  <td>
                    {attack ? (
                      <Tag tone="threat" icon="alert">
                        {f.prediction}
                      </Tag>
                    ) : (
                      <span className="faint">Benign</span>
                    )}
                  </td>
                  <td>
                    <span className="conf mono">
                      {pct(f.confidence, 0)}
                      <span className="conf__track">
                        <i style={{ width: `${f.confidence * 100}%` }} />
                      </span>
                    </span>
                  </td>
                  {hasTruth && (
                    <td>
                      {f.truth == null ? (
                        <span className="faint">{NA}</span>
                      ) : wrong ? (
                        <Tag tone="warn" title="The model disagrees with the label">
                          {f.truth}
                          <span className="sr-only"> (misclassified)</span>
                        </Tag>
                      ) : (
                        <span className="muted">{f.truth}</span>
                      )}
                    </td>
                  )}
                  {columns.map((c) => (
                    <td key={c.name} className={`num mono${c.wide ? " col-wide" : ""}`}>
                      {featureValue(f.x[c.index])}
                    </td>
                  ))}
                </tr>
              );
            })}
            {visible.length === 0 && (
              <tr>
                <td colSpan={4 + columns.length}>
                  <p className="dt__empty">No flows match this filter.</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </TableScroll>
      <div className="log-foot">
        <Pager page={current} pageSize={PAGE_SIZE} total={rows.length} onPage={setPage} />
        <p className="hint">Select a row to see the model's reasoning for that flow.</p>
      </div>
    </Panel>
  );
}
