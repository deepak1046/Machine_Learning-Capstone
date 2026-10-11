import { useRef, useState } from "react";
import { fileSize } from "../format";
import { Icon } from "./Icon";
import { Field, Segmented } from "./ui";

const PER_CLASS_IN_SAMPLE = 150;

type Source = "sample" | "upload";

interface Props {
  source: Source;
  onSource: (source: Source) => void;
  sampleN: number;
  onSampleN: (n: number) => void;
  allTypes: string[];
  attackTypes: string[];
  onAttackTypes: (types: string[]) => void;
  file: File | null;
  onFile: (file: File | null) => void;
  loading: boolean;
  onRun: () => void;
  onShuffle: () => void;
}

export function SourcePanel(p: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const maxSample = Math.max(50, PER_CLASS_IN_SAMPLE * p.attackTypes.length);
  const toggle = (type: string) =>
    p.onAttackTypes(
      p.attackTypes.includes(type) ? p.attackTypes.filter((t) => t !== type) : [...p.attackTypes, type],
    );

  const pick = (file: File | undefined) => file && p.onFile(file);
  const canRun = !p.loading && (p.source === "sample" ? p.attackTypes.length > 0 : p.file !== null);

  return (
    <section className="source" aria-labelledby="source-title">
      <h2 className="panel__title" id="source-title">
        Input
      </h2>
      <Segmented
        variant="tab"
        label="Flow source"
        value={p.source}
        onChange={p.onSource}
        options={[
          { value: "sample", label: "Held-out sample", controls: "source-panel" },
          { value: "upload", label: "Upload CSV", controls: "source-panel" },
        ]}
      />

      <div id="source-panel" role="tabpanel" aria-label={p.source === "sample" ? "Held-out sample" : "Upload CSV"} className="source__body">
        {p.source === "sample" ? (
          <>
            <p className="hint">
              Flows from the test split. The models never saw these behaviors in training, and each flow carries its true
              label.
            </p>
            <Field label="Number of flows" value={Math.min(p.sampleN, maxSample)}>
              <input
                type="range"
                min={50}
                max={maxSample}
                step={50}
                value={Math.min(p.sampleN, maxSample)}
                onChange={(e) => p.onSampleN(Number(e.target.value))}
              />
            </Field>
            <div className="field">
              <span className="field__label field__label--split">
                Traffic classes
                <span className="links">
                  <button type="button" className="link" onClick={() => p.onAttackTypes(p.allTypes)}>
                    All
                  </button>
                  <button type="button" className="link" onClick={() => p.onAttackTypes(["BENIGN"])}>
                    Benign only
                  </button>
                </span>
              </span>
              <div className="chips" role="group" aria-label="Traffic classes to draw flows from">
                {p.allTypes.map((type) => (
                  <button
                    key={type}
                    type="button"
                    className={`chip${p.attackTypes.includes(type) ? " is-on" : ""}`}
                    aria-pressed={p.attackTypes.includes(type)}
                    onClick={() => toggle(type)}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>
          </>
        ) : (
          <>
            <div
              className={`dropzone${dragging ? " is-dragging" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                pick(e.dataTransfer.files[0]);
              }}
            >
              <input
                ref={input}
                type="file"
                accept=".csv,text/csv"
                hidden
                data-testid="file-input"
                onChange={(e) => pick(e.target.files?.[0])}
              />
              {p.file ? (
                <div className="dropzone__file">
                  <Icon name="file" size={18} />
                  <span className="dropzone__name">
                    {p.file.name} <span className="faint mono">{fileSize(p.file.size)}</span>
                  </span>
                  <button
                    type="button"
                    className="btn btn--icon btn--sm btn--quiet"
                    aria-label="Remove file"
                    onClick={() => p.onFile(null)}
                  >
                    <Icon name="close" size={14} />
                  </button>
                </div>
              ) : (
                <button type="button" className="dropzone__target" onClick={() => input.current?.click()}>
                  <Icon name="upload" size={20} />
                  <strong>Drop a CSV here or browse</strong>
                  <span className="faint">CICFlowMeter format, as in CICDDoS2019</span>
                </button>
              )}
            </div>
            <p className="hint">
              The file needs the 60 model features under the dataset's column names. A <code>Label</code> column is
              optional and adds accuracy, per-class metrics and a confusion matrix. The first 100,000 rows are scored.
            </p>
            <a className="link link--icon" href="/api/sample/download">
              <Icon name="download" size={14} />
              Download the sample CSV to try it
            </a>
          </>
        )}
      </div>

      <div className="source__actions">
        <button type="button" className="btn btn--primary" disabled={!canRun} onClick={p.onRun}>
          {p.loading ? "Classifying" : "Classify flows"}
        </button>
        {p.source === "sample" && (
          <button
            type="button"
            className="btn"
            disabled={!canRun}
            onClick={p.onShuffle}
            title="Draw a different set of flows"
          >
            <Icon name="refresh" size={14} />
            Draw again
          </button>
        )}
      </div>
    </section>
  );
}
