import { useState } from "react";
import { ApiError, predict } from "../lib/api";
import CodeBlock from "../components/CodeBlock";
import FactorBars from "../components/FactorBars";
import RiskBadge from "../components/RiskBadge";
import RiskGauge from "../components/RiskGauge";
import { toCurl } from "../lib/curl";
import { PROGRAMS, type PredictionResponse, type StudentTermFeatures } from "../lib/types";

const DEFAULT_FEATURES: StudentTermFeatures = {
  term_number: 3,
  enrollment_intensity: "full_time",
  credits_attempted: 15,
  credits_completed: 12,
  credit_completion_rate: 0.8,
  cumulative_credits_attempted: 42,
  cumulative_credits_completed: 34,
  cumulative_credit_completion_rate: 0.81,
  term_gpa: 2.1,
  cumulative_gpa: 2.4,
  gpa_change: -0.3,
  academic_momentum: -0.25,
  courses_withdrawn: 1,
  cumulative_withdrawals: 2,
  courses_repeated: 0,
  cumulative_repeats: 1,
  advising_contact_flag: 0,
  financial_aid_flag: 1,
  prior_term_enrolled_flag: 1,
  program: "Business",
  entry_type: "first_time",
};

const inputClass =
  "mt-1.5 w-full rounded-lg border border-[var(--border-subtle)] bg-black/20 px-3 py-2 text-sm text-[var(--text-primary)] outline-none transition-colors focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]";
const labelClass = "block text-sm font-medium text-[var(--text-secondary)]";
const sectionTitleClass = "mb-1 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]";

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className={labelClass}>
      {label}
      <input
        type="number"
        className={inputClass}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function SliderField({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.05,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
}) {
  const progress = ((value - min) / (max - min)) * 100;
  return (
    <label className={labelClass}>
      <span className="flex items-baseline justify-between">
        {label}
        <span className="font-mono-ui text-xs text-[var(--accent-bright)]">{value.toFixed(2)}</span>
      </span>
      <input
        type="range"
        className="mt-2.5 w-full"
        style={{ "--range-progress": `${progress}%` } as React.CSSProperties}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function CheckField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="group flex cursor-pointer items-center gap-2.5 text-sm text-[var(--text-secondary)]">
      <input
        type="checkbox"
        className="peer sr-only"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span
        className="flex h-5 w-5 shrink-0 items-center justify-center rounded border border-[var(--border-strong)] bg-black/20 transition-colors
          peer-checked:border-[var(--accent)] peer-checked:bg-[var(--accent)]
          peer-focus-visible:ring-2 peer-focus-visible:ring-[var(--accent-soft)]"
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="opacity-0 peer-checked:[&]:opacity-100" style={{ opacity: checked ? 1 : 0 }}>
          <path d="M2 6l2.5 2.5L10 3" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
      {label}
    </label>
  );
}

export default function Predict() {
  const [features, setFeatures] = useState<StudentTermFeatures>(DEFAULT_FEATURES);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [requestSent, setRequestSent] = useState<StudentTermFeatures | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showInspector, setShowInspector] = useState(false);

  const set = <K extends keyof StudentTermFeatures>(key: K, value: StudentTermFeatures[K]) =>
    setFeatures((prev) => ({ ...prev, [key]: value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    const startedAt = performance.now();
    try {
      const response = await predict(features);
      setLatencyMs(performance.now() - startedAt);
      setRequestSent(features);
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Request failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-10">
      <div>
        <h1 className="font-display text-2xl font-bold text-[var(--text-primary)]">Student-level prediction</h1>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          Enter one student-term's engineered features (see the repo's{" "}
          <code className="font-mono-ui rounded bg-white/5 px-1 py-0.5">data/DATA_DICTIONARY.md</code> for field
          meanings).
        </p>
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="glass-panel space-y-4 p-5">
          <div className={sectionTitleClass}>Academic standing</div>
          <NumberField label="Term number" value={features.term_number} min={1} max={20} onChange={(v) => set("term_number", v)} />
          <label className={labelClass}>
            Enrollment intensity
            <select
              className={inputClass}
              value={features.enrollment_intensity}
              onChange={(e) => set("enrollment_intensity", e.target.value as StudentTermFeatures["enrollment_intensity"])}
            >
              <option value="full_time">Full time</option>
              <option value="part_time">Part time</option>
            </select>
          </label>
          <label className={labelClass}>
            Program
            <select
              className={inputClass}
              value={features.program}
              onChange={(e) => set("program", e.target.value as StudentTermFeatures["program"])}
            >
              {PROGRAMS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label className={labelClass}>
            Entry type
            <select
              className={inputClass}
              value={features.entry_type}
              onChange={(e) => set("entry_type", e.target.value as StudentTermFeatures["entry_type"])}
            >
              <option value="first_time">First time</option>
              <option value="transfer">Transfer</option>
            </select>
          </label>
          <SliderField label="Term GPA" value={features.term_gpa} min={0} max={4} onChange={(v) => set("term_gpa", v)} />
          <SliderField label="Cumulative GPA" value={features.cumulative_gpa} min={0} max={4} onChange={(v) => set("cumulative_gpa", v)} />
        </div>

        <div className="glass-panel space-y-4 p-5">
          <div className={sectionTitleClass}>Credits & momentum</div>
          <NumberField label="Credits attempted (this term)" value={features.credits_attempted} min={0} max={30} onChange={(v) => set("credits_attempted", v)} />
          <NumberField label="Credits completed (this term)" value={features.credits_completed} min={0} max={30} onChange={(v) => set("credits_completed", v)} />
          <NumberField label="Cumulative credits attempted" value={features.cumulative_credits_attempted} min={0} onChange={(v) => set("cumulative_credits_attempted", v)} />
          <NumberField label="Cumulative credits completed" value={features.cumulative_credits_completed} min={0} onChange={(v) => set("cumulative_credits_completed", v)} />
          <SliderField label="GPA change vs. recent terms" value={features.gpa_change} min={-4} max={4} onChange={(v) => set("gpa_change", v)} />
          <SliderField label="Academic momentum" value={features.academic_momentum} min={-4} max={4} onChange={(v) => set("academic_momentum", v)} />
        </div>

        <div className="glass-panel space-y-4 p-5">
          <div className={sectionTitleClass}>History & support</div>
          <NumberField label="Courses withdrawn (this term)" value={features.courses_withdrawn} min={0} max={10} onChange={(v) => set("courses_withdrawn", v)} />
          <NumberField label="Cumulative withdrawals" value={features.cumulative_withdrawals} min={0} onChange={(v) => set("cumulative_withdrawals", v)} />
          <NumberField label="Courses repeated (this term)" value={features.courses_repeated} min={0} max={10} onChange={(v) => set("courses_repeated", v)} />
          <NumberField label="Cumulative repeats" value={features.cumulative_repeats} min={0} onChange={(v) => set("cumulative_repeats", v)} />
          <div className="space-y-3 pt-1">
            <CheckField
              label="Had advising contact this term"
              checked={features.advising_contact_flag === 1}
              onChange={(v) => set("advising_contact_flag", v ? 1 : 0)}
            />
            <CheckField
              label="Receiving financial aid"
              checked={features.financial_aid_flag === 1}
              onChange={(v) => set("financial_aid_flag", v ? 1 : 0)}
            />
            <CheckField
              label="Enrolled in immediately prior term"
              checked={features.prior_term_enrolled_flag === 1}
              onChange={(v) => set("prior_term_enrolled_flag", v ? 1 : 0)}
            />
          </div>
        </div>

        <div className="md:col-span-3">
          <button
            type="submit"
            disabled={loading}
            className="rounded-full bg-[var(--accent)] px-6 py-2.5 text-sm font-medium text-white shadow-[0_0_24px_-4px_var(--accent-glow)] transition-transform hover:scale-[1.02] hover:bg-[var(--accent-bright)] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100"
          >
            {loading ? "Predicting…" : "Predict"}
          </button>
        </div>
      </form>

      {error && <div className="glass-panel border-[var(--risk-high)]/30 p-4 text-sm text-[var(--text-secondary)]">{error}</div>}

      {result && (
        <div className="glass-panel glow-ring space-y-5 p-6">
          <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-start">
            <RiskGauge persistenceProbability={result.persistence_probability} category={result.risk_category} />
            <div className="flex-1 space-y-3 text-center sm:text-left">
              <RiskBadge category={result.risk_category} />
              <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-[var(--text-secondary)] sm:justify-start">
                <span>
                  Risk probability:{" "}
                  <strong className="text-[var(--text-primary)]">{(result.risk_probability * 100).toFixed(1)}%</strong>
                </span>
                <span className="font-mono-ui text-xs text-[var(--text-muted)]">model {result.model_version}</span>
                {latencyMs !== null && (
                  <span className="font-mono-ui inline-flex items-center gap-1 text-xs text-[var(--text-muted)]">
                    <span className="h-1 w-1 rounded-full bg-[var(--cyan)]" />
                    {latencyMs.toFixed(0)}ms
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="border-t border-[var(--border-subtle)] pt-4">
            <h3 className="mb-3 text-sm font-semibold text-[var(--text-primary)]">Top contributing factors</h3>
            <FactorBars factors={result.top_contributing_factors} scoreScale={result.score_scale} />
          </div>

          <div className="border-t border-[var(--border-subtle)] pt-4">
            <button
              type="button"
              onClick={() => setShowInspector((v) => !v)}
              className="flex items-center gap-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:text-[var(--text-primary)]"
            >
              <span className={`inline-block transition-transform ${showInspector ? "rotate-90" : ""}`}>›</span>
              {showInspector ? "Hide" : "Show"} request/response details
            </button>
            {showInspector && requestSent && (
              <div className="mt-3 space-y-3">
                <CodeBlock label="curl" code={toCurl(requestSent)} />
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <CodeBlock label="Request body" code={JSON.stringify(requestSent, null, 2)} />
                  <CodeBlock label="Response body" code={JSON.stringify(result, null, 2)} />
                </div>
              </div>
            )}
          </div>

          <p className="border-t border-[var(--border-subtle)] pt-3 text-xs text-[var(--text-muted)]">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
