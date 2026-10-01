import { useState } from "react";
import { ApiError, predict } from "../lib/api";
import FactorBars from "../components/FactorBars";
import RiskBadge from "../components/RiskBadge";
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
    <label className="block text-sm">
      <span className="font-medium text-slate-700">{label}</span>
      <input
        type="number"
        className="mt-1 w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:border-slate-500 focus:outline-none"
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
  return (
    <label className="block text-sm">
      <span className="font-medium text-slate-700">
        {label}: <span className="font-normal text-slate-500">{value.toFixed(2)}</span>
      </span>
      <input
        type="range"
        className="mt-1 w-full accent-slate-900"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export default function Predict() {
  const [features, setFeatures] = useState<StudentTermFeatures>(DEFAULT_FEATURES);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const set = <K extends keyof StudentTermFeatures>(key: K, value: StudentTermFeatures[K]) =>
    setFeatures((prev) => ({ ...prev, [key]: value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predict(features));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Request failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Student-level prediction</h1>
        <p className="mt-1 text-sm text-slate-600">
          Enter one student-term's engineered features (see the repo's{" "}
          <code className="rounded bg-slate-100 px-1">data/DATA_DICTIONARY.md</code> for field meanings).
        </p>
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <div className="space-y-4 rounded-lg border border-slate-200 bg-white p-4">
          <NumberField label="Term number" value={features.term_number} min={1} max={20} onChange={(v) => set("term_number", v)} />
          <label className="block text-sm">
            <span className="font-medium text-slate-700">Enrollment intensity</span>
            <select
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
              value={features.enrollment_intensity}
              onChange={(e) => set("enrollment_intensity", e.target.value as StudentTermFeatures["enrollment_intensity"])}
            >
              <option value="full_time">Full time</option>
              <option value="part_time">Part time</option>
            </select>
          </label>
          <label className="block text-sm">
            <span className="font-medium text-slate-700">Program</span>
            <select
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
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
          <label className="block text-sm">
            <span className="font-medium text-slate-700">Entry type</span>
            <select
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
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

        <div className="space-y-4 rounded-lg border border-slate-200 bg-white p-4">
          <NumberField label="Credits attempted (this term)" value={features.credits_attempted} min={0} max={30} onChange={(v) => set("credits_attempted", v)} />
          <NumberField label="Credits completed (this term)" value={features.credits_completed} min={0} max={30} onChange={(v) => set("credits_completed", v)} />
          <NumberField label="Cumulative credits attempted" value={features.cumulative_credits_attempted} min={0} onChange={(v) => set("cumulative_credits_attempted", v)} />
          <NumberField label="Cumulative credits completed" value={features.cumulative_credits_completed} min={0} onChange={(v) => set("cumulative_credits_completed", v)} />
          <SliderField label="GPA change vs. recent terms" value={features.gpa_change} min={-4} max={4} onChange={(v) => set("gpa_change", v)} />
          <SliderField label="Academic momentum" value={features.academic_momentum} min={-4} max={4} onChange={(v) => set("academic_momentum", v)} />
        </div>

        <div className="space-y-4 rounded-lg border border-slate-200 bg-white p-4">
          <NumberField label="Courses withdrawn (this term)" value={features.courses_withdrawn} min={0} max={10} onChange={(v) => set("courses_withdrawn", v)} />
          <NumberField label="Cumulative withdrawals" value={features.cumulative_withdrawals} min={0} onChange={(v) => set("cumulative_withdrawals", v)} />
          <NumberField label="Courses repeated (this term)" value={features.courses_repeated} min={0} max={10} onChange={(v) => set("courses_repeated", v)} />
          <NumberField label="Cumulative repeats" value={features.cumulative_repeats} min={0} onChange={(v) => set("cumulative_repeats", v)} />
          <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
            <input
              type="checkbox"
              checked={features.advising_contact_flag === 1}
              onChange={(e) => set("advising_contact_flag", e.target.checked ? 1 : 0)}
            />
            Had advising contact this term
          </label>
          <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
            <input
              type="checkbox"
              checked={features.financial_aid_flag === 1}
              onChange={(e) => set("financial_aid_flag", e.target.checked ? 1 : 0)}
            />
            Receiving financial aid
          </label>
          <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
            <input
              type="checkbox"
              checked={features.prior_term_enrolled_flag === 1}
              onChange={(e) => set("prior_term_enrolled_flag", e.target.checked ? 1 : 0)}
            />
            Enrolled in immediately prior term
          </label>
        </div>

        <div className="md:col-span-3">
          <button
            type="submit"
            disabled={loading}
            className="rounded-md bg-slate-900 px-5 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
          >
            {loading ? "Predicting…" : "Predict"}
          </button>
        </div>
      </form>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">{error}</div>
      )}

      {result && (
        <div className="space-y-4 rounded-lg border border-slate-200 bg-white p-6">
          <div className="flex flex-wrap items-center gap-4">
            <RiskBadge category={result.risk_category} />
            <div className="text-sm text-slate-600">
              Persistence probability: <strong>{(result.persistence_probability * 100).toFixed(1)}%</strong>
            </div>
            <div className="text-sm text-slate-600">
              Risk probability: <strong>{(result.risk_probability * 100).toFixed(1)}%</strong>
            </div>
            <div className="text-xs text-slate-400">Model {result.model_version}</div>
          </div>
          <div>
            <h3 className="mb-2 text-sm font-semibold text-slate-700">Top contributing factors</h3>
            <FactorBars factors={result.top_contributing_factors} scoreScale={result.score_scale} />
          </div>
          <p className="border-t border-slate-100 pt-3 text-xs text-slate-400">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
