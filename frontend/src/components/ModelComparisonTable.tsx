import type { ModelMetrics } from "../lib/types";

const METRICS: { key: keyof ModelMetrics; label: string; higherIsBetter: boolean }[] = [
  { key: "roc_auc", label: "ROC-AUC", higherIsBetter: true },
  { key: "f1", label: "F1", higherIsBetter: true },
  { key: "precision", label: "Precision", higherIsBetter: true },
  { key: "recall", label: "Recall", higherIsBetter: true },
  { key: "brier_score", label: "Brier score", higherIsBetter: false },
];

export default function ModelComparisonTable({
  validationMetricsByModel,
  bestHyperparameters,
  selectedModel,
  ensembleMembers,
}: {
  validationMetricsByModel: Record<string, ModelMetrics>;
  bestHyperparameters: Record<string, Record<string, number | string | null>>;
  selectedModel: string;
  ensembleMembers: string[] | null;
}) {
  const models = Object.keys(validationMetricsByModel);

  const bestForMetric = (key: keyof ModelMetrics, higherIsBetter: boolean) => {
    const values = models.map((m) => Number(validationMetricsByModel[m][key]));
    return higherIsBetter ? Math.max(...values) : Math.min(...values);
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-0 text-sm">
        <thead>
          <tr>
            <th className="pb-2 text-left text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              Candidate (validation set)
            </th>
            {METRICS.map((m) => (
              <th key={m.key} className="pb-2 text-right text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((model) => {
            const isWinner = model === selectedModel;
            return (
              <tr key={model} className={isWinner ? "bg-[var(--accent-soft)]" : ""}>
                <td className="rounded-l-lg py-2.5 pl-3">
                  <div className="flex items-center gap-2">
                    {isWinner && <span className="text-[var(--accent-bright)]">★</span>}
                    <span className={`font-mono-ui ${isWinner ? "text-[var(--text-primary)]" : "text-[var(--text-secondary)]"}`}>
                      {model.replace(/_/g, " ")}
                    </span>
                  </div>
                  {model === "ensemble" && ensembleMembers && (
                    <div className="mt-0.5 text-xs text-[var(--text-muted)]">= {ensembleMembers.join(" + ")}</div>
                  )}
                  {bestHyperparameters[model] && (
                    <div className="font-mono-ui mt-0.5 text-[10px] text-[var(--text-muted)]">
                      {Object.entries(bestHyperparameters[model])
                        .map(([k, v]) => `${k}=${v}`)
                        .join(", ")}
                    </div>
                  )}
                </td>
                {METRICS.map((m) => {
                  const value = Number(validationMetricsByModel[model][m.key]);
                  const isBest = value === bestForMetric(m.key, m.higherIsBetter);
                  return (
                    <td
                      key={m.key}
                      className={`py-2.5 pr-3 text-right font-mono-ui ${
                        isBest ? "font-semibold text-[var(--risk-low)]" : "text-[var(--text-secondary)]"
                      }`}
                    >
                      {value.toFixed(3)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        ★ = selected model (best validation ROC-AUC). Green = best value per column. Every candidate shown, not
        just the winner — the ensemble only ships because it genuinely beat every individually-tuned model.
      </p>
    </div>
  );
}
