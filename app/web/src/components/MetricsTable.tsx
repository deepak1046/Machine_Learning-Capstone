import { useMemo, useState } from "react";
import { count, f3 } from "../format";
import type { Classification } from "../types";
import { SortTh, TableScroll } from "./ui";

type Key = "class" | "support" | "precision" | "recall" | "f1";

const WEAK = 0.5;

function Bar({ value }: { value: number }) {
  return (
    <span className="mini">
      <span className="mono">{f3(value)}</span>
      <span className="mini__track">
        <i className={value < WEAK ? "is-weak" : undefined} style={{ width: `${value * 100}%` }} />
      </span>
    </span>
  );
}

/** Precision, recall and F1 per class, from the run's labeled flows. */
export function MetricsTable({ result }: { result: Classification }) {
  const truth = result.truth;
  const [sort, setSort] = useState<{ key: Key; dir: "asc" | "desc" }>({ key: "class", dir: "asc" });

  const predicted = useMemo(
    () => new Map(result.prediction_counts.map((c) => [c.class, c.count])),
    [result.prediction_counts],
  );

  const rows = useMemo(() => {
    if (!truth) return [];
    const kept = truth.per_class.filter((c) => c.support > 0 || (predicted.get(c.class) ?? 0) > 0);
    const sign = sort.dir === "asc" ? 1 : -1;
    return [...kept].sort((a, b) =>
      sort.key === "class" ? sign * a.class.localeCompare(b.class) : sign * (a[sort.key] - b[sort.key]),
    );
  }, [truth, predicted, sort]);

  if (!truth) return null;

  const present = truth.per_class.filter((c) => c.support > 0);
  const mean = (pick: (c: (typeof present)[number]) => number) =>
    present.length ? present.reduce((sum, c) => sum + pick(c), 0) / present.length : null;

  const sortProps = (key: Key) => ({
    active: sort.key === key,
    dir: sort.dir,
    onSort: () => setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: key === "class" ? "asc" : "desc" })),
  });

  return (
    <TableScroll label="Per-class precision, recall and F1">
      <table className="dt">
        <thead>
          <tr>
            <SortTh label="Class" {...sortProps("class")} />
            <SortTh label="Support" numeric {...sortProps("support")} />
            <SortTh label="Precision" {...sortProps("precision")} />
            <SortTh label="Recall" {...sortProps("recall")} />
            <SortTh label="F1" {...sortProps("f1")} />
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.class}>
              <td>{c.class}</td>
              <td className="num mono">{count(c.support)}</td>
              {c.support === 0 ? (
                <td colSpan={3} className="faint">
                  Not in the labels, {count(predicted.get(c.class) ?? 0)} predicted
                </td>
              ) : (
                <>
                  <td>
                    <Bar value={c.precision} />
                  </td>
                  <td>
                    <Bar value={c.recall} />
                  </td>
                  <td>
                    <Bar value={c.f1} />
                  </td>
                </>
              )}
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <th scope="row">Macro average</th>
            <td className="num mono">{count(truth.n_labeled)}</td>
            <td className="mono">{f3(mean((c) => c.precision))}</td>
            <td className="mono">{f3(mean((c) => c.recall))}</td>
            <td className="mono">{f3(truth.macro_f1)}</td>
          </tr>
        </tfoot>
      </table>
    </TableScroll>
  );
}
