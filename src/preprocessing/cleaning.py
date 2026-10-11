from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The CICFlowMeter-generated columns that hold text, not numbers. Everything else in
# this dataset's schema is a numeric feature. Matched against stripped column names.
TEXT_COLUMNS = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "SimillarHTTP", "Label"}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip inconsistent leading/trailing whitespace from CICFlowMeter column names
    and drop the accidental `Unnamed: 0` index column some CSVs were saved with."""
    df = df.rename(columns=lambda c: c.strip())
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """CICFlowMeter emits inf/-inf on rate features (e.g. Flow Bytes/s, Flow Packets/s)
    for zero-duration flows. Converting to NaN makes them visible to isnull() checks
    and downstream imputation instead of silently poisoning stats like describe()."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _dtype_map_for(csv_file: Path) -> dict:
    """Fix every column's dtype from the header alone, instead of letting pandas
    infer it per chunk.

    Chunked reads infer dtype independently per chunk: a mostly-numeric text column
    like SimillarHTTP reads as float64 in a chunk that happens to contain only "0"s
    and as string in a chunk containing its rare non-numeric value, producing
    mismatched schemas between chunks that break streaming Parquet writes. Pinning
    every column's dtype up front from the header makes every chunk identical.
    """
    header = pd.read_csv(csv_file, nrows=0)
    return {
        col: ("string" if col.strip() in TEXT_COLUMNS else "float64")
        for col in header.columns
    }


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = clean_columns(chunk)
    return replace_inf_with_nan(chunk)


def process_file_in_chunks(
    src_file: Path,
    dest_file: Path,
    chunksize: int = 500_000,
    dedupe: bool = False,
) -> None:
    """Clean one CSV (column names, Unnamed: 0, inf/-inf) and write it to dest_file
    as compressed Parquet (dest_file should end in .parquet).

    Streams the file in chunks so it never holds more than `chunksize` rows in memory.
    `dedupe=True` loads the whole file at once to drop duplicate rows first -- only use
    it for files already confirmed small enough to fit in memory (see profile_df's
    "Duplicate Rows" / "Size (MB)" columns from 01_dataset_exploration.ipynb).
    """
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dtype_map = _dtype_map_for(src_file)

    if dedupe:
        df = pd.read_csv(src_file, dtype=dtype_map, low_memory=False)
        df = _clean_chunk(df)
        df = df.drop_duplicates()
        df.to_parquet(dest_file, index=False, compression="snappy")
        return

    writer = None
    try:
        for chunk in pd.read_csv(src_file, dtype=dtype_map, chunksize=chunksize, low_memory=False):
            chunk = _clean_chunk(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dest_file, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
