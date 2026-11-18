"""Step 2: ingestion. Read UCI Adult from OpenML and store an untouched raw snapshot.

fetch_openml(name='adult', version=2, as_frame=True) -> bunch.frame (14 features + label 'class').
Nothing is renamed, recoded, filtered or imputed here: the frame is written as-is to
data/lab04/raw/adult_raw.csv (missing values become empty fields) and its SHA256 is logged.
The row and column counts are logged and compared with the expected 48842 x 15.

Outputs: results/ingestion_log.json, data/lab04/raw/adult_raw.csv (git-ignored)
"""

from __future__ import annotations

from sklearn.datasets import fetch_openml

from common import CFG, OPENML_CACHE, RAW_FILE, RESULTS_DIR, set_seeds, sha256_file, write_csv, write_json


def main() -> None:
    set_seeds()
    ds = CFG["dataset"]
    bunch = fetch_openml(name=ds["name"], version=ds["version"], as_frame=True, data_home=str(OPENML_CACHE))
    frame = bunch.frame
    rows, cols = frame.shape
    write_csv(frame, RAW_FILE)
    details = getattr(bunch, "details", {}) or {}
    log = {
        "source": f"OpenML fetch_openml(name='{ds['name']}', version={ds['version']}, as_frame=True)",
        "openml_id": details.get("id"),
        "openml_name": details.get("name"),
        "openml_version": details.get("version"),
        "rows": rows,
        "columns": cols,
        "expected_rows": ds["expected_rows"],
        "expected_columns": ds["expected_columns"],
        "shape_matches_expected": rows == ds["expected_rows"] and cols == ds["expected_columns"],
        "column_names": list(frame.columns),
        "raw_dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "missing_values_per_column": {c: int(n) for c, n in frame.isna().sum().items()},
        "raw_snapshot": str(RAW_FILE.relative_to(RAW_FILE.parents[3])).replace("\\", "/"),
        "raw_snapshot_sha256": sha256_file(RAW_FILE),
        "modifications": "none - the frame is written exactly as returned (missing values as empty fields)",
    }
    write_json(RESULTS_DIR / "ingestion_log.json", log)
    print(f"Ingested {rows} rows x {cols} columns (expected {ds['expected_rows']} x {ds['expected_columns']}): "
          f"{'OK' if log['shape_matches_expected'] else 'MISMATCH'}")
    print(f"Raw snapshot {log['raw_snapshot']}  sha256 {log['raw_snapshot_sha256']}")


if __name__ == "__main__":
    main()
