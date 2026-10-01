import type { RiskCategory } from "../lib/types";

const RISK_STYLE: Record<RiskCategory, { stroke: string; glow: string; label: string }> = {
  low: { stroke: "#34d399", glow: "rgba(52, 211, 153, 0.45)", label: "Low risk" },
  medium: { stroke: "#fbbf24", glow: "rgba(251, 191, 36, 0.45)", label: "Medium risk" },
  high: { stroke: "#fb7185", glow: "rgba(251, 113, 133, 0.45)", label: "High risk" },
};

const SIZE = 168;
const STROKE = 10;
const RADIUS = (SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function RiskGauge({
  persistenceProbability,
  category,
}: {
  persistenceProbability: number;
  category: RiskCategory;
}) {
  const style = RISK_STYLE[category];
  const offset = CIRCUMFERENCE * (1 - persistenceProbability);

  return (
    <div className="relative flex flex-col items-center" style={{ width: SIZE, height: SIZE }}>
      <svg width={SIZE} height={SIZE} className="-rotate-90">
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          fill="none"
          stroke="var(--border-subtle)"
          strokeWidth={STROKE}
        />
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          fill="none"
          stroke={style.stroke}
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          style={{ filter: `drop-shadow(0 0 8px ${style.glow})`, transition: "stroke-dashoffset 0.6s ease-out" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-4xl font-semibold text-slate-50">
          {(persistenceProbability * 100).toFixed(0)}
          <span className="text-xl text-slate-400">%</span>
        </span>
        <span className="mt-1 text-xs uppercase tracking-wider text-slate-500">persistence</span>
      </div>
    </div>
  );
}
