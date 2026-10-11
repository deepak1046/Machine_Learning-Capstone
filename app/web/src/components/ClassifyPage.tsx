import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { classifySample, classifyUpload } from "../api";
import { NA, count, f3, modelName, pct, seconds } from "../format";
import { binaryView } from "../metrics";
import type { Classification, Families, ModelInfo } from "../types";
import { BarGroups } from "./Charts";
import { ConfusionMatrix } from "./ConfusionMatrix";
import { ExplainPanel } from "./ExplainPanel";
import { FlowTable } from "./FlowTable";
import { MetricsTable } from "./MetricsTable";
import { ModelPicker } from "./ModelPicker";
import { SourcePanel } from "./SourcePanel";
import { EmptyState, Notice, PageHeader, Panel, Readouts, Spinner, type ReadoutItem } from "./ui";

interface Props {
  models: ModelInfo[];
  families: Families;
  active: boolean;
}

export function ClassifyPage({ models, families, active }: Props) {
  const [modelKey, setModelKey] = useState("xgb13");
  const [source, setSource] = useState<"sample" | "upload">("sample");
  const [sampleN, setSampleN] = useState(500);
  const [seed, setSeed] = useState(0);
  const allTypes = useMemo(() => models.find((m) => m.key === "xgb13")?.classes ?? [], [models]);
  const [attackTypes, setAttackTypes] = useState<string[] | null>(null);
  const [file, setFile] = useState<File | null>(null);

  const [result, setResult] = useState<Classification | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const latest = useRef(0);
  const detailPane = useRef<HTMLDivElement>(null);
  const tableAnchor = useRef<HTMLDivElement>(null);

  const chosenTypes = attackTypes ?? allTypes;
  const model = models.find((m) => m.key === modelKey) ?? models[0];
  const familyModel = models.find((m) => m.level === "attack family");

  const run = useCallback(
    async (next: { model?: string; seed?: number } = {}) => {
      const key = next.model ?? modelKey;
      const request = ++latest.current;
      if (source === "upload" && !file) {
        setError("Choose a CSV file first.");
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const res =
          source === "sample"
            ? await classifySample({
                model: key,
                n: Math.min(sampleN, Math.max(50, 150 * chosenTypes.length)),
                seed: next.seed ?? seed,
                classes: chosenTypes.length === allTypes.length ? null : chosenTypes,
              })
            : await classifyUpload(key, file!);
        if (request !== latest.current) return;
        setResult(res);
        setSelectedId(res.flows.length > 0 ? 0 : null);
      } catch (e) {
        if (request === latest.current) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (request === latest.current) setLoading(false);
      }
    },
    [modelKey, source, file, sampleN, seed, chosenTypes, allTypes.length],
  );

  // the first sample runs when the page is first opened
  const started = useRef(false);
  useEffect(() => {
    if (active && !started.current && allTypes.length > 0) {
      started.current = true;
      void run();
    }
  }, [active, allTypes.length, run]);

  const switchModel = (key: string) => {
    setModelKey(key);
    if (source === "sample" || file) void run({ model: key });
  };

  const groupOf = (attack: string) => {
    const members = Object.values(families).find((group) => group.includes(attack));
    return members ?? null;
  };

  const selectFlow = (id: number) => {
    setSelectedId(id);
    const pane = detailPane.current;
    // when the detail sits below the table, bring it into view
    if (pane && pane.getBoundingClientRect().top > window.innerHeight - 120) {
      const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      requestAnimationFrame(() => pane.scrollIntoView({ block: "start", behavior: calm ? "auto" : "smooth" }));
    }
  };

  const resultModel = result ? (models.find((m) => m.key === result.model) ?? model) : model;
  const selectedFlow = result?.flows.find((f) => f.id === selectedId) ?? null;
  const binary = result ? binaryView(result) : null;

  const classRows = useMemo(() => {
    if (!result) return [];
    const predicted = new Map(result.prediction_counts.map((c) => [c.class, c.count]));
    const support = new Map(result.truth?.per_class.map((c) => [c.class, c.support]) ?? []);
    return result.classes
      .filter((c) => (predicted.get(c) ?? 0) > 0 || (support.get(c) ?? 0) > 0)
      .sort((a, b) => (predicted.get(b) ?? 0) - (predicted.get(a) ?? 0))
      .map((c) => ({
        label: c,
        values: result.truth ? [predicted.get(c) ?? 0, support.get(c) ?? 0] : [predicted.get(c) ?? 0],
      }));
  }, [result]);
  const maxCount = Math.max(1, ...classRows.flatMap((r) => r.values.map((v) => v ?? 0)));

  const groups = useMemo<{ run: ReadoutItem[]; labels: ReadoutItem[] | null }>(() => {
    if (!result) return { run: [], labels: null };
    const truth = result.truth;
    const attacks = Math.round(result.attack_share * result.n_flows);
    const run: ReadoutItem[] = [
      {
        label: "Flows scored",
        value: count(result.n_flows),
        sub: result.truncated ? "first 100,000 rows of the file" : seconds(result.elapsed_ms),
      },
      { label: "Flagged as attack", value: pct(result.attack_share), sub: `${count(attacks)} of ${count(result.n_flows)} flows` },
      {
        label: "Mean confidence",
        value: pct(result.mean_confidence, 0),
        sub: "model score",
        hint: "The model's score after class-balanced training. It is not a calibrated probability.",
      },
    ];
    if (!truth) {
      run.push({ label: "Labels", value: "None", sub: "no Label column in this data" });
      return { run, labels: null };
    }
    const labels: ReadoutItem[] = [
      { label: "Accuracy", value: pct(truth.accuracy), sub: `${count(truth.n_labeled)} labeled flows` },
      {
        label: "Macro-F1",
        value: f3(truth.macro_f1),
        sub: `${truth.per_class.filter((c) => c.support > 0).length} classes in the labels`,
      },
    ];
    if (binary) {
      labels.push(
        {
          label: "Detection rate",
          value: binary.detection == null ? NA : pct(binary.detection),
          sub: `${count(binary.caught)} of ${count(binary.attacks)} attack flows`,
          hint: "Attack flows flagged as any attack class, whether or not the type was right",
        },
        {
          label: "False positive rate",
          value: binary.falsePositive == null ? NA : pct(binary.falsePositive),
          sub: `${count(binary.falseAlarms)} of ${count(binary.benign)} benign flows`,
          hint: "Benign flows flagged as an attack class",
        },
      );
    }
    return { run, labels };
  }, [result, binary]);

  return (
    <div className="page">
      <PageHeader
        title="Attack Classification"
        description="Score flows with a trained model, compare each prediction with the true label, and see which features drove it."
      />

      <div className="panel config">
        <ModelPicker models={models} value={modelKey} onChange={switchModel} />
        <SourcePanel
          source={source}
          onSource={setSource}
          sampleN={sampleN}
          onSampleN={setSampleN}
          allTypes={allTypes}
          attackTypes={chosenTypes}
          onAttackTypes={setAttackTypes}
          file={file}
          onFile={setFile}
          loading={loading}
          onRun={() => void run()}
          onShuffle={() => {
            const next = seed + 1;
            setSeed(next);
            void run({ seed: next });
          }}
        />
      </div>

      <div className="results" aria-busy={loading}>
        {error && (
          <Notice tone="threat" title="Classification failed">
            {error}
          </Notice>
        )}
        {loading && (
          <p className="busy" role="status">
            <Spinner />
            Classifying with {model ? modelName(model) : "the model"}
            {model?.kind === "neural" && !model.loaded ? ". The network is still loading, this takes a few seconds." : ""}
          </p>
        )}

        {!result && !loading && !error && (
          <EmptyState icon="crosshair" title="No results yet">
            Choose a model and an input, then press Classify flows.
          </EmptyState>
        )}

        {result && (
          <div className={`results__body${loading ? " is-stale" : ""}`}>
            <div className={`readout-groups${groups.labels ? "" : " readout-groups--single"}`}>
              <div className="readout-group">
                <h2 className="readout-group__title">
                  {result.source === "sample" ? "Held-out sample" : "Uploaded file"} scored by {modelName(resultModel)}
                </h2>
                <Readouts label="Run summary" items={groups.run} />
              </div>
              {groups.labels && (
                <div className="readout-group">
                  <h2 className="readout-group__title">Against the true labels</h2>
                  <Readouts label="Metrics against the true labels" items={groups.labels} />
                </div>
              )}
            </div>

            <div className={`layout layout--pair${result.truth ? "" : " layout--single"}`}>
              <Panel title={result.truth ? "Predicted and actual classes" : "Predicted classes"} meta={<span className="faint">flows per class</span>}>
                <BarGroups
                  rows={classRows}
                  series={
                    result.truth
                      ? [
                          { name: "predicted", color: "var(--c-accent)" },
                          { name: "actual", color: "var(--c-series-b)" },
                        ]
                      : [{ name: "predicted", color: "var(--c-accent)" }]
                  }
                  max={maxCount * 1.12}
                  showAxis={false}
                  labelWidth={84}
                  format={count}
                  labelRows={classRows.map((r) => r.label)}
                  ariaLabel="Number of flows per predicted class"
                />
              </Panel>
              {result.truth && (
                <Panel title="Confusion matrix">
                  <ConfusionMatrix classes={result.classes} truth={result.truth} />
                </Panel>
              )}
            </div>

            {result.truth && (
              <Panel title="Per-class metrics" meta={<span className="faint">against the true labels</span>} flush>
                <MetricsTable result={result} />
              </Panel>
            )}

            <div className="layout layout--flows">
              <div ref={tableAnchor} className="layout__main">
                <FlowTable result={result} selectedId={selectedId} onSelect={selectFlow} />
              </div>
              <div ref={detailPane} className="layout__side">
                <ExplainPanel
                  model={resultModel}
                  flow={selectedFlow}
                  groupOf={groupOf}
                  familyModel={familyModel}
                  onUseFamilyModel={() => familyModel && switchModel(familyModel.key)}
                  onBack={() => tableAnchor.current?.scrollIntoView({ block: "start", behavior: "smooth" })}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
