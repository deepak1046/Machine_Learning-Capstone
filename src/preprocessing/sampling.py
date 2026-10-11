from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def stratified_sample_parquet(
    src_file: Path,
    class_counts: dict,
    class_col: str = "Label",
    cap_per_class: int = 100_000,
    batch_size: int = 500_000,
    seed: int = 42,
    columns: list | None = None,
) -> pd.DataFrame:
    """Stream `src_file` once and keep a Bernoulli-sampled subset of each class,
    capped at `cap_per_class` rows (classes smaller than the cap are kept whole).

    `class_counts` must be the exact total row count per class (e.g. from
    reports/class_distribution.csv) so the per-class keep-probability is correct --
    Parquet metadata doesn't expose this without a full scan.

    Never loads the source file whole: only the kept subset (bounded by roughly
    cap_per_class * n_classes) accumulates in memory.
    """
    rng = np.random.default_rng(seed)
    keep_frac = {cls: min(1.0, cap_per_class / count) for cls, count in class_counts.items()}

    kept_chunks = []
    for batch in pq.ParquetFile(src_file).iter_batches(batch_size=batch_size, columns=columns):
        chunk = batch.to_pandas()
        frac = chunk[class_col].map(keep_frac).to_numpy(dtype="float64")
        keep_mask = rng.random(len(chunk)) < frac
        if keep_mask.any():
            kept_chunks.append(chunk.loc[keep_mask])

    return pd.concat(kept_chunks, ignore_index=True)
