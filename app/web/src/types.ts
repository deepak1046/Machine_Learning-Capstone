export type Level = "attack type" | "attack family";

export interface Heldout {
  macro_f1: number | null;
  accuracy: number | null;
  ceiling_macro_f1: number | null;
  ceiling_accuracy: number | null;
}

export interface ModelInfo {
  key: string;
  title: string;
  kind: "tree" | "neural";
  level: Level;
  classes: string[];
  blurb: string;
  explainable: boolean;
  loaded: boolean;
  heldout: Heldout;
}

export interface Alternative {
  class: string;
  p: number;
}

export interface Flow {
  id: number;
  prediction: string;
  confidence: number;
  truth: string | null;
  top: Alternative[];
  x: (number | null)[];
}

export interface PerClass {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface TruthScore {
  n_labeled: number;
  n_skipped: number;
  accuracy: number;
  macro_f1: number;
  per_class: PerClass[];
  confusion: number[][];
}

export interface Classification {
  model: string;
  level: Level;
  source: "sample" | "upload";
  classes: string[];
  n_flows: number;
  truncated: boolean;
  prediction_counts: { class: string; count: number }[];
  attack_share: number;
  mean_confidence: number;
  truth: TruthScore | null;
  feature_names: string[];
  flows: Flow[];
  elapsed_ms: number;
}

export interface Contribution {
  feature: string;
  value: number | null;
  contribution: number;
}

export interface Explanation {
  predicted: string;
  confidence: number;
  alternatives: Alternative[];
  contributions: Contribution[];
  other_features: number;
  bias: number;
}

export interface PerClassTable {
  classes: string[];
  columns: string[];
  values: (number | null)[][];
}

export interface ResultsData {
  ceilings: { task: string; classes: number; rows: number; macro_f1: number; accuracy: number; cross_label_share: number }[];
  thirteen_class: {
    model: string;
    macro_f1_random: number | null;
    macro_f1_group: number | null;
    accuracy_random: number | null;
    accuracy_group: number | null;
    pct_ceiling_random: number | null;
    pct_ceiling_group: number | null;
    minutes_random: number | null;
    minutes_group: number | null;
  }[];
  family: {
    split: string;
    macro_f1: number;
    accuracy: number;
    ceiling_macro_f1: number;
    ceiling_accuracy: number;
    vs_ceiling: number;
  }[];
  base_paper: {
    split: string;
    model: string;
    accuracy: number;
    macro_f1: number | null;
    ceiling_accuracy: number;
    ceiling_macro_f1: number;
  }[];
  per_class_13: PerClassTable;
  per_class_family: PerClassTable;
}

export type Families = Record<string, string[]>;

export interface StreamFlow {
  prediction: string;
  confidence: number;
  truth: string | null;
  protocol: number | null;
  packets_per_s: number | null;
  bytes_per_s: number | null;
  duration_us: number | null;
}

export interface StreamTick {
  model: string;
  level: Level;
  n_flows: number;
  flows: StreamFlow[];
  elapsed_ms: number;
}
