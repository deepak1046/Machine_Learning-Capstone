import { PageHeader, Section } from "./ui";

const DATASET: [string, string][] = [
  ["Source", "CICDDoS2019, flows extracted with CICFlowMeter"],
  ["Capture files", "18 files from two capture days"],
  ["Flows after cleaning", "About 70 million"],
  ["Features", "80+ CICFlowMeter features per flow, 60 non-redundant ones used by the models"],
  ["Classes", "13: BENIGN plus DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, UDP, Syn, TFTP, UDPLag and Portmap"],
  ["Dropped", "WebDDoS (439 flows), too rare to learn from"],
  ["Training sample", "Class-balanced, about 3.6 million flows"],
];

const PAGES: [string, string][] = [
  ["Traffic Replay", "Replays held-out flows once per second through the selected model and shows the verdicts as they would appear in a monitor. A simulation."],
  ["Classification", "Scores a held-out sample or an uploaded CSV, compares predictions with the labels and explains single flows."],
  ["Model Evaluation", "Reads every trained model's held-out scores against the behavior-only ceiling."],
];

const LIMITS = [
  "Trained on one lab-generated dataset: a demonstration, not an intrusion detection system.",
  "The replay is not captured traffic. Its attacks, their type and their volume are scenario settings.",
  "Confidence is the model's score after class-balanced training, not a calibrated probability.",
  "Feature-level explanations exist for the tree models only. The neural network has none.",
  "Uploads are scored up to the first 100,000 rows.",
];

export function AboutPage() {
  return (
    <div className="page page--reading">
      <PageHeader
        title="Dataset and Method"
        description="What the models were trained on, how they were tested, and what limits their scores."
      />

      <Section title="Purpose">
        <div className="prose">
          <p>
            This is the front end of a machine-learning capstone on DDoS detection. It classifies network flows from the
            CICDDoS2019 dataset into twelve attack types plus benign traffic or, with the family model, into eight broader
            groups. It uses the project's trained XGBoost models and the proposed CNN-BiLSTM-Attention network.
          </p>
        </div>
      </Section>

      <Section title="Dataset">
        <dl className="facts">
          {DATASET.map(([term, detail]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{detail}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section title="How the models were tested">
        <div className="prose">
          <p>
            Flows that are identical in every feature except ports are kept together, so the held-out test set contains
            behaviors the models have never seen. The models served here were trained on that split, and the held-out sample
            in the app comes from its test side. The scores you see are an honest estimate of how the models generalize.
          </p>
        </div>
      </Section>

      <Section title="Why the 13-class score tops out near 0.81">
        <div className="prose">
          <p>
            Some attack types produce byte-for-byte identical flows, because an amplified response fragment looks the same
            whichever protocol was abused. No model can separate what the data does not separate. The Model Evaluation page
            shows how each model compares with that ceiling, and how grouping the confusable types into families removes the
            problem.
          </p>
        </div>
      </Section>

      <Section title="Pages">
        <dl className="facts">
          {PAGES.map(([term, detail]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{detail}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section title="Limits">
        <ul className="bullets">
          {LIMITS.map((limit) => (
            <li key={limit}>{limit}</li>
          ))}
        </ul>
      </Section>

      <Section title="Built with">
        <div className="prose">
          <p>
            Python (pandas, XGBoost, Keras and TensorFlow, FastAPI) for the models and the API. React, TypeScript and Vite for
            this interface, set in IBM Plex.
          </p>
        </div>
      </Section>
    </div>
  );
}
