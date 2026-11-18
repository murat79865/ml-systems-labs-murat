"""Export every Lab 6 MLflow run (experiments lab06-*) to results/mlflow_runs.csv - a readable record of
the experiment tracking. The full store is lab06/mlflow_runs/; browse it with
    mlflow ui --backend-store-uri lab06/mlflow_runs
"""

from __future__ import annotations

import mlflow

from common import RESULTS_DIR, load_config, repo_path, write_json

KEEP_PARAMS = ["data.batch_size", "data.num_workers", "optimizer.name", "optimizer.lr", "scheduler.name",
               "grad_accum_steps", "epochs", "checkpoint.stop_after_epoch"]
KEEP_METRICS = ["test_acc", "val_acc", "train_loss", "throughput_samples_per_s", "share_data_loading",
                "share_forward", "share_backward", "share_optimizer_step", "train_samples_per_s", "data_wait_share"]


def main() -> None:
    cfg = load_config()
    mlflow.set_tracking_uri(repo_path(cfg["mlflow"]["tracking_dir"]).resolve().as_uri())
    exps = [e for e in mlflow.search_experiments() if e.name.startswith("lab06-")]
    df = mlflow.search_runs(experiment_ids=[e.experiment_id for e in exps])
    names = {e.experiment_id: e.name for e in exps}
    df["experiment"] = df["experiment_id"].map(names)
    cols = ["experiment", "tags.mlflow.runName", "status", "run_id"]
    cols += [f"params.{p}" for p in KEEP_PARAMS if f"params.{p}" in df.columns]
    cols += [f"metrics.{m}" for m in KEEP_METRICS if f"metrics.{m}" in df.columns]
    out = df[cols].sort_values(["experiment", "tags.mlflow.runName"])
    out.columns = [c.replace("tags.mlflow.runName", "run_name").replace("params.", "").replace("metrics.", "")
                   for c in out.columns]
    out.to_csv(RESULTS_DIR / "mlflow_runs.csv", index=False, lineterminator="\n", float_format="%.5f")
    summary = {"experiments": sorted(names.values()), "runs": int(len(out)),
               "runs_per_experiment": out["experiment"].value_counts().sort_index().to_dict()}
    write_json(RESULTS_DIR / "mlflow_summary.json", summary)
    print(f"MLflow: {summary['runs']} runs in {len(exps)} experiments -> results/mlflow_runs.csv")
    for k, v in summary["runs_per_experiment"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
