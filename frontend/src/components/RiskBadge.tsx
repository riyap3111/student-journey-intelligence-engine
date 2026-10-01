import type { RiskCategory } from "../lib/types";

const STYLES: Record<RiskCategory, string> = {
  low: "bg-emerald-100 text-emerald-800 ring-emerald-600/20",
  medium: "bg-amber-100 text-amber-800 ring-amber-600/20",
  high: "bg-red-100 text-red-800 ring-red-600/20",
};

export default function RiskBadge({ category }: { category: RiskCategory }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset ${STYLES[category]}`}
    >
      {category.toUpperCase()} RISK
    </span>
  );
}
