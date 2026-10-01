import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, getDriftReport } from "../lib/api";
import StatCard from "../components/StatCard";
import type { DriftReportResponse, FeatureDriftDetail } from "../lib/types";

const STATUS_COLOR: Record<FeatureDriftDetail["status"], string> = {
  stable: "#34d399",
  moderate_shift: "#fbbf24",
  significant_shift: "#fb7185",
  insufficient_data: "#5f6b81",
};

const STATUS_LABEL: Record<FeatureDriftDetail["status"], string> = {
  stable: "Stable",
  moderate_shift: "Moderate shift",
  significant_shift: "Significant shift",
  insufficient_data: "Insufficient data",
};

const GRID_COLOR = "rgba(148, 163, 184, 0.1)";

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
      <div className="mx-auto max-w-5xl px-4 py-10">
        <div className="glass-panel border-[var(--risk-high)]/30 p-4 text-sm text-[var(--text-secondary)]">{error}</div>
      </div>
    );
  }

  if (!report) {
    return <div className="mx-auto max-w-5xl px-4 py-10 text-[var(--text-secondary)]">Loading drift report…</div>;
  }

  const chartData = report.features.map((f) => ({ ...f, psi: f.psi ?? 0 }));

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-10">
      <div>
        <h1 className="font-display text-2xl font-bold text-[var(--text-primary)]">Feature Drift Monitoring</h1>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
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
          accent="cyan"
        />
        <StatCard label="Min samples required" value={`${report.min_samples_required}`} />
      </div>

      <div className="glass-panel p-5">
        <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          PSI by feature
        </h2>
        <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 28)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 24, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11, fill: "#5f6b81" }} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
            <YAxis
              type="category"
              dataKey="feature"
              width={200}
              tick={{ fontSize: 11, fill: "#94a3b8" }}
              axisLine={{ stroke: GRID_COLOR }}
              tickLine={false}
            />
            <ReferenceLine x={0.1} stroke="#fbbf24" strokeDasharray="4 4" />
            <ReferenceLine x={0.25} stroke="#fb7185" strokeDasharray="4 4" />
            <Tooltip
              contentStyle={{ background: "#11131f", border: "1px solid rgba(148, 163, 184, 0.2)", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#f1f5f9" }}
              itemStyle={{ color: "#94a3b8" }}
              formatter={(value) => Number(value).toFixed(4)}
            />
            <Bar dataKey="psi" radius={3}>
              {chartData.map((entry) => (
                <Cell key={entry.feature} fill={STATUS_COLOR[entry.status]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-2 text-xs text-[var(--text-muted)]">
          Dashed lines mark the conventional PSI thresholds: 0.10 (moderate shift) and 0.25 (significant
          shift).
        </p>
      </div>

      <p className="text-xs text-[var(--text-muted)]">{report.disclaimer}</p>
    </div>
  );
}
