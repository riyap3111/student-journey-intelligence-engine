import { useEffect, useState } from "react";
import { ApiError, getModelInfo } from "../lib/api";
import StatCard from "../components/StatCard";
import type { ModelInfoResponse } from "../lib/types";

const METRIC_LABELS: Record<string, string> = {
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  roc_auc: "ROC-AUC",
  average_precision: "Avg. precision",
  brier_score: "Brier score",
};

export default function ModelInfo() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Could not reach the API."));
  }, []);

  if (error) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-10">
        <div className="glass-panel border-[var(--risk-high)]/30 p-4 text-sm text-[var(--text-secondary)]">{error}</div>
      </div>
    );
  }

  if (!info) {
    return <div className="mx-auto max-w-5xl px-4 py-10 text-[var(--text-secondary)]">Loading model info…</div>;
  }

  const metricEntries = Object.entries(info.test_metrics).filter(([k]) => k in METRIC_LABELS);

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-10">
      <h1 className="font-display text-2xl font-bold text-[var(--text-primary)]">Model Info</h1>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Model type" value={info.model_type.replace("_", " ")} />
        <StatCard label="Version" value={info.model_version} accent="cyan" />
        <StatCard label="Low risk max" value={info.risk_thresholds.low_max.toFixed(3)} />
        <StatCard label="Medium risk max" value={info.risk_thresholds.medium_max.toFixed(3)} accent="cyan" />
      </div>

      <div className="glass-panel p-5">
        <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Held-out test metrics
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {metricEntries.map(([key, value]) => (
            <StatCard key={key} label={METRIC_LABELS[key]} value={Number(value).toFixed(3)} />
          ))}
        </div>
      </div>

      <div className="glass-panel p-5">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Feature columns ({info.feature_columns.length})
        </h2>
        <div className="flex flex-wrap gap-1.5">
          {info.feature_columns.map((f) => (
            <span
              key={f}
              className="font-mono-ui rounded-md border border-[var(--border-subtle)] bg-white/[0.03] px-2 py-1 text-xs text-[var(--text-secondary)]"
            >
              {f}
            </span>
          ))}
        </div>
      </div>

      <div className="glass-panel border-[var(--risk-medium)]/20 p-5">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-[var(--risk-medium)]">
          Excluded demographic proxy columns (never used as model inputs)
        </h2>
        <div className="flex flex-wrap gap-1.5">
          {info.excluded_demographic_proxy_columns.map((f) => (
            <span
              key={f}
              className="font-mono-ui rounded-md border border-[var(--risk-medium)]/20 bg-[var(--risk-medium-soft)] px-2 py-1 text-xs text-[var(--risk-medium)]"
            >
              {f}
            </span>
          ))}
        </div>
      </div>

      <p className="text-xs text-[var(--text-muted)]">
        Trained: {new Date(info.trained_at_utc).toLocaleString()}. {info.disclaimer}
      </p>
    </div>
  );
}
