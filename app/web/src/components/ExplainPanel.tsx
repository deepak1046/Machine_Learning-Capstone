import { useEffect, useRef, useState } from "react";
import { explain } from "../api";
import { featureValue, pct } from "../format";
import type { Explanation, Flow, ModelInfo } from "../types";
import { ContribBars } from "./Charts";
import { Notice, Panel, Spinner, Tag } from "./ui";

interface Props {
  model: ModelInfo;
  flow: Flow | null;
  groupOf: (attack: string) => string[] | null;
  familyModel: ModelInfo | undefined;
  onUseFamilyModel: () => void;
  /** Scrolls back to the flow table; only shown when the detail sits below the table. */
  onBack?: () => void;
}

export function ExplainPanel({ model, flow, groupOf, familyModel, onUseFamilyModel, onBack }: Props) {
  const cache = useRef(new Map<string, Explanation>());
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!flow || !model.explainable) {
      setExplanation(null);
      return;
    }
    const key = `${model.key}|${flow.x.join("|")}`;
    const hit = cache.current.get(key);
    if (hit) {
      setExplanation(hit);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    explain(model.key, flow.x)
      .then((result) => {
        if (cancelled) return;
        cache.current.set(key, result);
        setExplanation(result);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [flow, model.key, model.explainable]);

  if (!flow) {
    return (
      <Panel title="Prediction detail">
        <p className="hint">
          Select a flow in the table to see the model's top candidates and, for the XGBoost models, the features behind
          the score.
        </p>
      </Panel>
    );
  }

  const attack = flow.prediction !== "BENIGN";
  const group = model.level === "attack type" ? groupOf(flow.prediction) : null;
  const wrong = flow.truth != null && flow.truth !== flow.prediction;

  return (
    <Panel
      title="Prediction detail"
      meta={
        <>
          {onBack && (
            <button type="button" className="link only-stacked" onClick={onBack}>
              Back to the table
            </button>
          )}
          <span className="faint mono">Flow {flow.id + 1}</span>
        </>
      }
    >
      <div className="detail" aria-live="polite">
        <div className="detail__head">
          <Tag tone={attack ? "threat" : "neutral"} icon={attack ? "alert" : undefined}>
            {flow.prediction}
          </Tag>
          <strong className="mono">{pct(flow.confidence, 0)}</strong>
          <span className="faint">model score</span>
          {flow.truth != null &&
            (wrong ? (
              <Tag tone="warn" icon="alert">
                True label: {flow.truth}
              </Tag>
            ) : (
              <Tag tone="accent" icon="check">
                Matches the true label
              </Tag>
            ))}
        </div>

        <h3 className="detail__sub">Top candidates</h3>
        <div className="alts">
          {flow.top.map((alt) => (
            <div className="alt" key={alt.class}>
              <span>{alt.class}</span>
              <span className="alt__track">
                <i style={{ width: `${alt.p * 100}%` }} />
              </span>
              <span className="mono num">{pct(alt.p, 0)}</span>
            </div>
          ))}
        </div>

        {group && (
          <Notice tone="accent" title="These attack types overlap">
            {group.join(", ")} flows are often identical in flow statistics, so no model can reliably separate them.
            {familyModel && (
              <>
                {" "}
                The family model treats them as one group
                {familyModel.heldout.macro_f1 != null && ` (held-out macro-F1 ${familyModel.heldout.macro_f1.toFixed(3)})`}.{" "}
                <button type="button" className="link" onClick={onUseFamilyModel}>
                  Switch to the family model
                </button>
              </>
            )}
          </Notice>
        )}

        {!model.explainable && (
          <p className="hint">
            Feature-level explanations are available for the XGBoost models. The neural network has no built-in
            attribution, so only its top candidates are shown.
          </p>
        )}
        {model.explainable && loading && !explanation && (
          <p className="detail__loading">
            <Spinner /> Computing feature contributions
          </p>
        )}
        {error && (
          <Notice tone="threat" title="Explanation failed">
            {error}
          </Notice>
        )}
        {explanation && (
          <>
            <h3 className="detail__sub">Feature contributions</h3>
            <ContribBars
              target={explanation.predicted}
              items={[
                ...explanation.contributions.map((c) => ({
                  label: c.feature,
                  detail: `value ${featureValue(c.value)}`,
                  value: c.contribution,
                })),
                { label: "All other features", detail: "", value: explanation.other_features },
              ]}
            />
            <p className="hint">
              TreeSHAP contributions in log-odds: how far each feature moved the model's score for {explanation.predicted}.
              Together with a base score of {explanation.bias.toFixed(2)} (which includes the model's class offset, if it has one) they add up to the model's decision score.
            </p>
          </>
        )}
      </div>
    </Panel>
  );
}
