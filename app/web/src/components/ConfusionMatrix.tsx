import { useContext } from "react";
import { ThemeContext, heatColor } from "../theme";
import type { TruthScore } from "../types";
import { Heatmap } from "./Charts";

export function ConfusionMatrix({ classes, truth }: { classes: string[]; truth: TruthScore }) {
  const theme = useContext(ThemeContext);
  const rows = truth.confusion;
  const totals = rows.map((row) => row.reduce((a, b) => a + b, 0));
  // exact zeros stay empty so the cells that matter stand out
  const shares = rows.map((row, r) => row.map((v) => (totals[r] && v > 0 ? v / totals[r] : null)));

  return (
    <div className="confusion">
      <div className="legend">
        <span>
          <i className="legend__swatch legend__swatch--accent" />
          Correct (diagonal)
        </span>
        <span>
          <i className="legend__swatch legend__swatch--warn" />
          Misclassified
        </span>
      </div>
      <Heatmap
        rows={classes}
        cols={classes.map((c) => ({ top: c }))}
        values={shares}
        verticalHeaders
        corner="Actual"
        cellText={(v) => (v >= 0.005 ? String(Math.round(v * 100)) : "")}
        cellStyle={(v, r, c) => (r === c ? heatColor(v, theme, "accent") : heatColor(Math.sqrt(v), theme, "warn"))}
        cellTitle={(_, r, c) =>
          totals[r] === 0
            ? `${classes[r]}: no flows with this label`
            : `${rows[r][c].toLocaleString("en")} ${classes[r]} flows predicted as ${classes[c]} (${((rows[r][c] / totals[r]) * 100).toFixed(1)}% of ${classes[r]})`
        }
        ariaLabel="Confusion matrix. Each row shows where the flows of one actual class were predicted."
      />
      <p className="hint">Rows are the actual class, columns the predicted class. Cells show the percent of each row.</p>
    </div>
  );
}
