import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ContributingFactor, ScoreScale } from "../lib/types";

const RISK_COLOR = "#d03b3b";
const PERSIST_COLOR = "#0ca30c";

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
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis type="category" dataKey="feature" width={180} tick={{ fontSize: 12 }} />
          <Tooltip
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
      <p className="mt-2 text-xs text-slate-400">
        Contribution values are on the model's own {scoreScale === "log_odds" ? "log-odds" : "probability"} scale
        (detected per model type, not assumed).
      </p>
    </div>
  );
}
