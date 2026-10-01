import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, getDriftReport } from "../lib/api";
import StatCard from "../components/StatCard";
import type { DriftReportResponse, FeatureDriftDetail } from "../lib/types";

const STATUS_COLOR: Record<FeatureDriftDetail["status"], string> = {
  stable: "#0ca30c",
  moderate_shift: "#fab219",
  significant_shift: "#d03b3b",
  insufficient_data: "#9aa0a6",
};

const STATUS_LABEL: Record<FeatureDriftDetail["status"], string> = {
  stable: "Stable",
  moderate_shift: "Moderate shift",
  significant_shift: "Significant shift",
  insufficient_data: "Insufficient data",
};

export default function Monitoring() {
  const [report, setReport] = useState<DriftReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDriftReport()
      .then(setReport)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Could not reach the API."));
  }, []);

  if (error) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">{error}</div>
      </div>
    );
  }

  if (!report) {
    return <div className="mx-auto max-w-5xl px-4 py-8 text-slate-500">Loading drift report…</div>;
  }

  const chartData = report.features.map((f) => ({ ...f, psi: f.psi ?? 0 }));

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Feature Drift Monitoring</h1>
        <p className="mt-1 text-sm text-slate-600">
          Population Stability Index (PSI) comparing recently-scored API requests against the training
          distribution. Below {report.min_samples_required} logged requests, every feature reports
          "insufficient data" rather than a misleadingly precise number.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <StatCard label="Overall status" value={STATUS_LABEL[report.overall_status]} />
        <StatCard
          label="Requests compared"
          value={`${report.n_current_rows}`}
          caption={`of ${report.n_reference_rows ?? "?"} training rows`}
        />
        <StatCard label="Min samples required" value={`${report.min_samples_required}`} />
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-3 text-sm font-semibold text-slate-700">PSI by feature</h2>
        <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 28)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 24, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" width={200} tick={{ fontSize: 11 }} />
            <ReferenceLine x={0.1} stroke="#fab219" strokeDasharray="4 4" />
            <ReferenceLine x={0.25} stroke="#d03b3b" strokeDasharray="4 4" />
            <Tooltip formatter={(value) => Number(value).toFixed(4)} />
            <Bar dataKey="psi" radius={3}>
              {chartData.map((entry) => (
                <Cell key={entry.feature} fill={STATUS_COLOR[entry.status]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-2 text-xs text-slate-400">
          Dashed lines mark the conventional PSI thresholds: 0.10 (moderate shift) and 0.25 (significant
          shift).
        </p>
      </div>

      <p className="text-xs text-slate-400">{report.disclaimer}</p>
    </div>
  );
}
