import { useEffect, useMemo, useState } from "react";
import { getResults } from "../api";
import { NA, count, f3, modelName, pct } from "../format";
import type { ModelInfo, PerClassTable, ResultsData } from "../types";
import { BarGroups, Heatmap } from "./Charts";
import { EmptyState, Led, Notice, PageHeader, Panel, Readouts, Section, SortTh, Tag, TableScroll } from "./ui";

let cached: ResultsData | null = null;

const split = (name: string) => {
  const [top, bottom] = name.split(" | ");
  return { top, bottom };
};

function PerClassHeatmap({ table, label }: { table: PerClassTable; label: string }) {
  return (
    <Heatmap
      rows={table.classes}
      cols={table.columns.map(split)}
      values={table.values}
      gapAfterCol={0}
      maxWidth={150 + table.columns.length * 118}
      ariaLabel={label}
    />
  );
}

const PROTOCOLS = ["paper's reported result", "ours: paper-style split", "ours: random split", "ours: group split"];

type ModelRow = ResultsData["thirteen_class"][number];
type SortKey = "model" | "macro_f1_random" | "macro_f1_group" | "accuracy_random" | "accuracy_group" | "pct_ceiling_group" | "minutes_random";

const COLUMNS: { key: SortKey; label: string; numeric: boolean; hint?: string }[] = [
  { key: "model", label: "Model", numeric: false },
  { key: "macro_f1_random", label: "Macro-F1, random split", numeric: true },
  { key: "macro_f1_group", label: "Macro-F1, group split", numeric: true },
  { key: "accuracy_random", label: "Accuracy, random split", numeric: true },
  { key: "accuracy_group", label: "Accuracy, group split", numeric: true },
  { key: "pct_ceiling_group", label: "Group split, % of ceiling", numeric: true },
  { key: "minutes_random", label: "Minutes to train, random split", numeric: true },
];

function ModelComparison({ rows }: { rows: ModelRow[] }) {
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "macro_f1_group", dir: "desc" });
  const sorted = useMemo(() => {
    const sign = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      if (sort.key === "model") return sign * a.model.localeCompare(b.model);
      const x = a[sort.key];
      const y = b[sort.key];
      if (x == null && y == null) return 0;
      if (x == null) return 1; // runs that were not made always sort last
      if (y == null) return -1;
      return sign * (x - y);
    });
  }, [rows, sort]);

  return (
    <TableScroll label="Model comparison on the 13-class task">
      <table className="dt">
        <thead>
          <tr>
            {COLUMNS.map((c) => (
              <SortTh
                key={c.key}
                label={c.label}
                numeric={c.numeric}
                active={sort.key === c.key}
                dir={sort.dir}
                onSort={() =>
                  setSort((s) => (s.key === c.key ? { key: c.key, dir: s.dir === "asc" ? "desc" : "asc" } : { key: c.key, dir: c.key === "model" ? "asc" : "desc" }))
                }
              />
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.model}>
              <td>{r.model}</td>
              <td className="num mono">{f3(r.macro_f1_random)}</td>
              <td className="num mono">{f3(r.macro_f1_group)}</td>
              <td className="num mono">{f3(r.accuracy_random)}</td>
              <td className="num mono">{f3(r.accuracy_group)}</td>
              <td className="num mono">{r.pct_ceiling_group == null ? NA : `${r.pct_ceiling_group.toFixed(1)}%`}</td>
              <td className="num mono">{r.minutes_random == null ? NA : r.minutes_random.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableScroll>
  );
}

function ServedModels({ models }: { models: ModelInfo[] }) {
  return (
    <TableScroll label="Models served by the backend">
      <table className="dt">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Task</th>
            <th scope="col">Status</th>
            <th scope="col" className="num">
              Held-out macro-F1
            </th>
            <th scope="col" className="num">
              Held-out accuracy
            </th>
            <th scope="col" className="num">
              Ceiling macro-F1
            </th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.key}>
              <td>{modelName(m)}</td>
              <td className="muted">
                {m.classes.length} {m.level === "attack type" ? "attack types" : "attack families"}
              </td>
              <td>
                <span className="state">
                  <Led tone={m.loaded ? "ok" : "warn"} />
                  {m.loaded ? "Loaded" : "Loading"}
                </span>
              </td>
              <td className="num mono">{f3(m.heldout.macro_f1)}</td>
              <td className="num mono">{f3(m.heldout.accuracy)}</td>
              <td className="num mono">{f3(m.heldout.ceiling_macro_f1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableScroll>
  );
}

export function ResultsPage({ models }: { models: ModelInfo[] }) {
  const [data, setData] = useState<ResultsData | null>(cached);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (cached) return;
    getResults()
      .then((r) => {
        cached = r;
        setData(r);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const header = (
    <PageHeader
      title="Model Evaluation"
      description="Scores on held-out data for every trained model, measured against the best result that flow behavior alone allows."
    />
  );

  if (error) {
    return (
      <div className="page">
        {header}
        <Notice tone="threat" title="Evaluation results could not be loaded">
          {error}
        </Notice>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="page">
        {header}
        <EmptyState busy title="Loading evaluation results">
          Reading the saved reports from the backend.
        </EmptyState>
      </div>
    );
  }

  const ceiling13 = data.ceilings.find((c) => c.classes === 13);
  const ceiling8 = data.ceilings.find((c) => c.classes === 8 && c.rows > 1e6);
  const xgb = data.thirteen_class.find((r) => r.model === "XGBoost");
  const net = data.thirteen_class.find((r) => r.model === "CNN-BiLSTM-Attention (log + ports)");
  const best = (key: "macro_f1_random" | "macro_f1_group") =>
    [...data.thirteen_class].sort((a, b) => (b[key] ?? 0) - (a[key] ?? 0))[0];

  const paperAccuracy = (protocol: string, model: string) =>
    data.base_paper.find((r) => r.split === protocol && r.model === model)?.accuracy ?? null;
  const paperCeiling = data.base_paper[0]?.ceiling_accuracy;

  return (
    <div className="page">
      {header}

      <Panel title="Served models" meta={<span className="faint">held-out scores are on the group split</span>} flush>
        <ServedModels models={models} />
      </Panel>

      <Section title="Behavior-only ceiling">
        <div className="prose">
          <p>
            Many flows in CICDDoS2019 are identical in every feature except their ports, yet carry different labels. The same
            1,472-byte UDP fragment shows up under LDAP, DNS, SNMP and MSSQL alike. Identical inputs get identical
            predictions, so a model that ignores ports can do no better than giving each group of identical flows its most
            common label. That score is the behavior-only ceiling, and every result below is read against it.
          </p>
        </div>
        <TableScroll label="Behavior-only ceilings">
          <table className="dt">
            <thead>
              <tr>
                <th scope="col">Task</th>
                <th scope="col" className="num">
                  Classes
                </th>
                <th scope="col" className="num">
                  Flows
                </th>
                <th scope="col" className="num">
                  Ceiling macro-F1
                </th>
                <th scope="col" className="num">
                  Ceiling accuracy
                </th>
                <th scope="col" className="num">
                  Flows sharing behavior with another label
                </th>
              </tr>
            </thead>
            <tbody>
              {data.ceilings.map((c) => (
                <tr key={c.task}>
                  <td>{c.task}</td>
                  <td className="num mono">{c.classes}</td>
                  <td className="num mono">{count(c.rows)}</td>
                  <td className="num mono">{f3(c.macro_f1)}</td>
                  <td className="num mono">{f3(c.accuracy)}</td>
                  <td className="num mono">{pct(c.cross_label_share, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableScroll>
        <p className="hint">
          Random split: a stratified split of the flows. Group split: flows with identical behavior always land on the same
          side, so the test set holds behaviors the model never saw.
        </p>
      </Section>

      <Section title="13 attack types">
        {ceiling13 && xgb && net && (
          <Readouts
            label="Headline results on the group split"
            items={[
              {
                label: "XGBoost, group split",
                value: f3(xgb.macro_f1_group),
                sub: `${pct((xgb.pct_ceiling_group ?? 0) / 100, 0)} of the ${f3(ceiling13.macro_f1)} ceiling`,
              },
              {
                label: "CNN-BiLSTM-Attention, group split",
                value: f3(net.macro_f1_group),
                sub: `${pct((net.pct_ceiling_group ?? 0) / 100, 0)} of the ceiling`,
              },
              {
                label: "Best random split",
                value: f3(best("macro_f1_random").macro_f1_random),
                sub: best("macro_f1_random").model,
              },
            ]}
          />
        )}
        <Panel level={3} title="Macro-F1 by model" meta={<span className="faint">higher is better</span>}>
          {ceiling13 && (
            <BarGroups
              rows={data.thirteen_class.map((r) => ({ label: r.model, values: [r.macro_f1_random, r.macro_f1_group] }))}
              series={[
                { name: "random split", color: "var(--c-series-b)" },
                { name: "group split", color: "var(--c-accent)" },
              ]}
              ceiling={ceiling13.macro_f1}
              ceilingLabel="behavior-only ceiling"
              labelRows={[best("macro_f1_random").model, "CNN-BiLSTM-Attention (log + ports)"]}
              labelWidth={230}
              ariaLabel="Macro-F1 of each model on the 13-class task, random and group split, against the ceiling"
            />
          )}
        </Panel>
        <Panel level={3} title="Model comparison" meta={<span className="faint">select a column heading to sort</span>} flush>
          <ModelComparison rows={data.thirteen_class} />
        </Panel>
        <p className="hint">
          {NA} means the run was not made: the boosted-tree tuning only completed the random split.
        </p>
      </Section>

      <Section title="Per-class weaknesses">
        <div className="prose">
          <p>
            F1 per class for the 13-class models. On the group split, DNS, LDAP, Portmap, SSDP and UDP are weak for every
            model, the classes the ceiling marks as inherently confusable.
          </p>
        </div>
        <Panel level={3} title="F1 by class">
          <PerClassHeatmap table={data.per_class_13} label="Per-class F1 for the 13-class models and the ceiling" />
        </Panel>
      </Section>

      <Section title="8 attack families">
        <div className="prose">
          <p>
            Merging the confusable attack types (DNS, LDAP and SNMP; NetBIOS and Portmap; SSDP, UDP and UDPLag) removes the
            label ambiguity at its source and lifts the ceiling to {f3(ceiling8?.macro_f1)}. The same XGBoost then scores the
            first two rows below. The last three rows add a tuned decision rule: per-class offsets fitted on a validation
            slice of the training data, which re-balances the predictions that class-balanced training skews.
          </p>
        </div>
        <Panel level={3} title="Scores on 8 families" flush>
          <TableScroll label="Family model scores">
            <table className="dt">
              <thead>
                <tr>
                  <th scope="col">Split</th>
                  <th scope="col" className="num">
                    Macro-F1
                  </th>
                  <th scope="col" className="num">
                    Accuracy
                  </th>
                  <th scope="col" className="num">
                    Ceiling macro-F1
                  </th>
                  <th scope="col" className="num">
                    Ceiling accuracy
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.family.map((r) => (
                  <tr key={r.split}>
                    <td>{r.split}</td>
                    <td className="num mono">{f3(r.macro_f1)}</td>
                    <td className="num mono">{f3(r.accuracy)}</td>
                    <td className="num mono">{f3(r.ceiling_macro_f1)}</td>
                    <td className="num mono">{f3(r.ceiling_accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableScroll>
        </Panel>
        <p className="hint">
          A model that sees ports can score above this ceiling, and the group split does. The ceiling bounds what flow behavior
          alone can tell apart, not the model.
        </p>
        <Panel level={3} title="F1 by family">
          <PerClassHeatmap table={data.per_class_family} label="Per-class F1 for the 8-family XGBoost and its ceiling" />
        </Panel>
      </Section>

      <Section title="Reproducing the base paper">
        <div className="prose">
          <p>
            The base paper reports 99.55% accuracy for a one-epoch CNN on an 8-class subset of the 03-11 capture day. The same
            architecture and data, evaluated three ways, lands near 85%. That is below the {paperCeiling ? f3(paperCeiling) : ""}{" "}
            behavior-only ceiling for the subset, while the reported figure is above it.
          </p>
        </div>
        <Panel level={3} title="Accuracy of the base paper's models" meta={<Tag>8-class subset</Tag>}>
          <BarGroups
            rows={PROTOCOLS.map((p) => ({ label: p, values: [paperAccuracy(p, "CNN"), paperAccuracy(p, "LSTM")] }))}
            series={[
              { name: "CNN", color: "var(--c-accent)" },
              { name: "LSTM", color: "var(--c-series-b)" },
            ]}
            max={1.08}
            ceiling={paperCeiling}
            ceilingLabel="behavior-only ceiling"
            format={(v) => v.toFixed(4)}
            labelRows={PROTOCOLS}
            labelWidth={190}
            ariaLabel="Accuracy of the base paper's CNN and LSTM as reported and as reproduced under three splits"
          />
        </Panel>
        <p className="hint">The paper's LSTM figure (about 96.5%) is approximate.</p>
      </Section>
    </div>
  );
}
