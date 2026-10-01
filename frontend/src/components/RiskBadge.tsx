import type { RiskCategory } from "../lib/types";

const STYLES: Record<RiskCategory, { color: string; bg: string; border: string; dot: string }> = {
  low: { color: "#34d399", bg: "rgba(52, 211, 153, 0.1)", border: "rgba(52, 211, 153, 0.3)", dot: "#34d399" },
  medium: { color: "#fbbf24", bg: "rgba(251, 191, 36, 0.1)", border: "rgba(251, 191, 36, 0.3)", dot: "#fbbf24" },
  high: { color: "#fb7185", bg: "rgba(251, 113, 133, 0.1)", border: "rgba(251, 113, 133, 0.3)", dot: "#fb7185" },
};

export default function RiskBadge({ category }: { category: RiskCategory }) {
  const s = STYLES[category];
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-sm font-medium"
      style={{ color: s.color, background: s.bg, border: `1px solid ${s.border}` }}
    >
      <span className="h-1.5 w-1.5 rounded-full animate-pulse-glow" style={{ background: s.dot }} />
      {category.toUpperCase()} RISK
    </span>
  );
}
