import type { Classification } from "./types";

export interface BinaryView {
  attacks: number;
  benign: number;
  caught: number;
  falseAlarms: number;
  /** Share of attack flows flagged as any attack class. */
  detection: number | null;
  /** Share of benign flows flagged as an attack class. */
  falsePositive: number | null;
}

const sum = (row: number[]) => row.reduce((a, b) => a + b, 0);

/** Attack versus benign, derived from the confusion matrix the backend returned for this run. */
export function binaryView(result: Classification): BinaryView | null {
  const truth = result.truth;
  const benign = result.classes.indexOf("BENIGN");
  if (!truth || benign < 0) return null;

  const benignTotal = sum(truth.confusion[benign]);
  const falseAlarms = benignTotal - truth.confusion[benign][benign];
  let attacks = 0;
  let caught = 0;
  truth.confusion.forEach((row, r) => {
    if (r === benign) return;
    const total = sum(row);
    attacks += total;
    caught += total - row[benign];
  });

  return {
    attacks,
    benign: benignTotal,
    caught,
    falseAlarms,
    detection: attacks ? caught / attacks : null,
    falsePositive: benignTotal ? falseAlarms / benignTotal : null,
  };
}
