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
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">{error}</div>
      </div>
    );
  }

  if (!info) {
    return <div className="mx-auto max-w-5xl px-4 py-8 text-slate-500">Loading model info…</div>;
  }

  const metricEntries = Object.entries(info.test_metrics).filter(([k]) => k in METRIC_LABELS);

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      <h1 className="text-2xl font-bold text-slate-900">Model Info</h1>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Model type" value={info.model_type.replace("_", " ")} />
        <StatCard label="Version" value={info.model_version} />
        <StatCard label="Low risk max" value={info.risk_thresholds.low_max.toFixed(3)} />
        <StatCard label="Medium risk max" value={info.risk_thresholds.medium_max.toFixed(3)} />
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-3 text-sm font-semibold text-slate-700">Held-out test metrics</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {metricEntries.map(([key, value]) => (
            <StatCard key={key} label={METRIC_LABELS[key]} value={Number(value).toFixed(3)} />
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-2 text-sm font-semibold text-slate-700">
          Feature columns ({info.feature_columns.length})
        </h2>
        <div className="flex flex-wrap gap-1.5">
          {info.feature_columns.map((f) => (
            <span key={f} className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-700">
              {f}
            </span>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
        <h2 className="mb-2 text-sm font-semibold text-amber-900">
          Excluded demographic proxy columns (never used as model inputs)
        </h2>
        <div className="flex flex-wrap gap-1.5">
          {info.excluded_demographic_proxy_columns.map((f) => (
            <span key={f} className="rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-800">
              {f}
            </span>
          ))}
        </div>
      </div>

      <p className="text-xs text-slate-400">
        Trained: {new Date(info.trained_at_utc).toLocaleString()}. {info.disclaimer}
      </p>
    </div>
  );
}
