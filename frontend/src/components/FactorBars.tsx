import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ContributingFactor, ScoreScale } from "../lib/types";

const RISK_COLOR = "#fb7185";
const PERSIST_COLOR = "#34d399";
const GRID_COLOR = "rgba(148, 163, 184, 0.1)";
const AXIS_COLOR = "#5f6b81";

export default function FactorBars({
  factors,
  scoreScale,
}: {
  factors: ContributingFactor[];
  scoreScale: ScoreScale;
}) {
  const data = [...factors]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .map((f) => ({ ...f, absContribution: Math.abs(f.contribution) }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={Math.max(160, data.length * 44)}>
        <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11, fill: AXIS_COLOR }} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
          <YAxis
            type="category"
            dataKey="feature"
            width={180}
            tick={{ fontSize: 12, fill: "#94a3b8" }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "#11131f",
              border: "1px solid rgba(148, 163, 184, 0.2)",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#f1f5f9" }}
            itemStyle={{ color: "#94a3b8" }}
            formatter={(value, _name, props) => [
              Number(value).toFixed(4),
              props.payload.direction === "increases_risk" ? "Increases risk" : "Increases persistence",
            ]}
          />
          <Bar dataKey="contribution" radius={4}>
            {data.map((entry) => (
              <Cell
                key={entry.feature}
                fill={entry.direction === "increases_risk" ? RISK_COLOR : PERSIST_COLOR}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-[var(--text-muted)]">
        Contribution values are on the model's own {scoreScale === "log_odds" ? "log-odds" : "probability"} scale
        (detected per model type, not assumed).
      </p>
    </div>
  );
}
