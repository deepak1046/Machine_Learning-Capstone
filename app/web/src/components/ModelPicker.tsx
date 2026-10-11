import { f3 } from "../format";
import type { Heldout, ModelInfo } from "../types";
import { Spinner, Tag } from "./ui";

function ScoreMeter({ heldout }: { heldout: Heldout }) {
  if (heldout.macro_f1 == null) return null;
  const ceiling = heldout.ceiling_macro_f1;
  return (
    <div className="meter" title="Macro-F1 on held-out behaviors. The tick marks the behavior-only ceiling.">
      <div className="meter__text">
        <span>
          Held-out macro-F1 <strong className="mono">{f3(heldout.macro_f1)}</strong>
        </span>
        {ceiling != null && <span className="faint mono">ceiling {f3(ceiling)}</span>}
      </div>
      <div className="meter__track">
        <div className="meter__fill" style={{ width: `${heldout.macro_f1 * 100}%` }} />
        {ceiling != null && <div className="meter__tick" style={{ left: `${ceiling * 100}%` }} />}
      </div>
    </div>
  );
}

interface Props {
  models: ModelInfo[];
  value: string;
  onChange: (key: string) => void;
}

export function ModelPicker({ models, value, onChange }: Props) {
  return (
    <fieldset className="models">
      <legend className="panel__title">Model</legend>
      {models.map((m) => (
        <label key={m.key} className={`model${value === m.key ? " is-on" : ""}`}>
          <input
            type="radio"
            name="model"
            value={m.key}
            checked={value === m.key}
            onChange={() => onChange(m.key)}
            className="model__radio"
          />
          <span className="model__mark" aria-hidden="true" />
          <span className="model__body">
            <span className="model__head">
              <span className="model__title">{m.title}</span>
              <Tag>
                {m.classes.length} {m.level === "attack type" ? "attack types" : "attack families"}
              </Tag>
            </span>
            <span className="model__blurb">{m.blurb}</span>
            <ScoreMeter heldout={m.heldout} />
            {m.kind === "neural" && !m.loaded && (
              <span className="model__loading">
                <Spinner /> Loading the network
              </span>
            )}
          </span>
        </label>
      ))}
    </fieldset>
  );
}
