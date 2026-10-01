import type { ConfusionMatrix as ConfusionMatrixType } from "../lib/types";

function Cell({ label, value, total, tone }: { label: string; value: number; total: number; tone: "correct" | "wrong" }) {
  const intensity = total > 0 ? value / total : 0;
  const bg =
    tone === "correct"
      ? `rgba(52, 211, 153, ${0.08 + intensity * 0.35})`
      : `rgba(251, 113, 133, ${0.08 + intensity * 0.35})`;
  const border = tone === "correct" ? "rgba(52, 211, 153, 0.3)" : "rgba(251, 113, 133, 0.3)";
  return (
    <div
      className="flex flex-col items-center justify-center rounded-lg p-4"
      style={{ background: bg, border: `1px solid ${border}` }}
    >
      <span className="font-display text-2xl font-semibold text-[var(--text-primary)]">{value}</span>
      <span className="mt-0.5 text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{label}</span>
    </div>
  );
}

export default function ConfusionMatrix({ matrix }: { matrix: ConfusionMatrixType }) {
  const total = matrix.tn + matrix.fp + matrix.fn + matrix.tp;
  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-xs text-[var(--text-muted)]">
        <div />
        <div className="text-center">Predicted: not persist</div>
        <div className="text-center">Predicted: persist</div>

        <div className="flex items-center justify-center text-center">Actual: not persist</div>
        <Cell label="True negative" value={matrix.tn} total={total} tone="correct" />
        <Cell label="False positive" value={matrix.fp} total={total} tone="wrong" />

        <div className="flex items-center justify-center text-center">Actual: persist</div>
        <Cell label="False negative" value={matrix.fn} total={total} tone="wrong" />
        <Cell label="True positive" value={matrix.tp} total={total} tone="correct" />
      </div>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        {total.toLocaleString()} held-out test rows. Diagonal (green) = correct; off-diagonal (red) = errors.
      </p>
    </div>
  );
}
