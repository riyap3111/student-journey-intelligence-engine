// Mirrors the Pydantic schemas in src/student_journey/api/schemas.py.
// Kept in sync by hand since the frontend is a separate package from the
// Python backend — see README for the single source of truth.

export const PROGRAMS = [
  "Business",
  "Engineering",
  "Social Sciences",
  "Biology",
  "Computer Science",
  "Undeclared",
] as const;

export type Program = (typeof PROGRAMS)[number];
export type EnrollmentIntensity = "full_time" | "part_time";
export type EntryType = "first_time" | "transfer";
export type RiskCategory = "low" | "medium" | "high";
export type ScoreScale = "log_odds" | "probability";

export interface StudentTermFeatures {
  term_number: number;
  enrollment_intensity: EnrollmentIntensity;
  credits_attempted: number;
  credits_completed: number;
  credit_completion_rate: number;
  cumulative_credits_attempted: number;
  cumulative_credits_completed: number;
  cumulative_credit_completion_rate: number;
  term_gpa: number;
  cumulative_gpa: number;
  gpa_change: number;
  academic_momentum: number;
  courses_withdrawn: number;
  cumulative_withdrawals: number;
  courses_repeated: number;
  cumulative_repeats: number;
  advising_contact_flag: 0 | 1;
  financial_aid_flag: 0 | 1;
  prior_term_enrolled_flag: 0 | 1;
  program: Program;
  entry_type: EntryType;
}

export interface ContributingFactor {
  feature: string;
  contribution: number;
  direction: "increases_persistence_likelihood" | "increases_risk";
}

export interface PredictionResponse {
  persistence_probability: number;
  risk_probability: number;
  risk_category: RiskCategory;
  score_scale: ScoreScale;
  top_contributing_factors: ContributingFactor[];
  model_version: string;
  disclaimer: string;
}

export interface HealthResponse {
  status: "ok";
  model_loaded: boolean;
  model_version: string | null;
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface ModelMetrics {
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  average_precision: number;
  brier_score: number;
  confusion_matrix: ConfusionMatrix;
  n_samples: number;
  positive_rate: number;
}

export interface ModelInfoResponse {
  model_version: string;
  model_type: string;
  trained_at_utc: string;
  feature_columns: string[];
  excluded_demographic_proxy_columns: string[];
  risk_thresholds: { low_max: number; medium_max: number };
  test_metrics: ModelMetrics;
  test_metrics_uncalibrated: ModelMetrics;
  validation_metrics_by_model: Record<string, ModelMetrics>;
  best_hyperparameters: Record<string, Record<string, number | string | null>>;
  ensemble_members: string[] | null;
  disclaimer: string;
}

export interface FeatureDriftDetail {
  feature: string;
  feature_type: "numeric" | "categorical";
  psi: number | null;
  status: "stable" | "moderate_shift" | "significant_shift" | "insufficient_data";
}

export interface DriftReportResponse {
  overall_status: FeatureDriftDetail["status"];
  n_reference_rows: number | null;
  n_current_rows: number;
  min_samples_required: number;
  features: FeatureDriftDetail[];
  disclaimer: string;
}

export interface ApiErrorBody {
  detail: string | { msg: string; loc: (string | number)[] }[];
}
