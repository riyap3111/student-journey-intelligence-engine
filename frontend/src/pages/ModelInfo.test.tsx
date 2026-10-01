import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ModelInfo from "./ModelInfo";

const MOCK_RESPONSE = {
  model_version: "20260101-000000",
  model_type: "ensemble",
  trained_at_utc: "2026-01-01T00:00:00Z",
  feature_columns: ["term_gpa", "program"],
  excluded_demographic_proxy_columns: ["age_band"],
  risk_thresholds: { low_max: 0.3, medium_max: 0.5 },
  test_metrics: {
    precision: 0.8, recall: 0.9, f1: 0.85, roc_auc: 0.7, average_precision: 0.82,
    brier_score: 0.18, confusion_matrix: { tn: 10, fp: 5, fn: 3, tp: 40 },
    n_samples: 58, positive_rate: 0.74,
  },
  test_metrics_uncalibrated: {
    precision: 0.78, recall: 0.88, f1: 0.83, roc_auc: 0.7, average_precision: 0.8,
    brier_score: 0.21, confusion_matrix: { tn: 9, fp: 6, fn: 4, tp: 39 },
    n_samples: 58, positive_rate: 0.74,
  },
  validation_metrics_by_model: {
    logistic_regression: {
      precision: 0.75, recall: 0.85, f1: 0.8, roc_auc: 0.68, average_precision: 0.78,
      brier_score: 0.22, confusion_matrix: { tn: 8, fp: 7, fn: 5, tp: 38 }, n_samples: 58, positive_rate: 0.74,
    },
    random_forest: {
      precision: 0.76, recall: 0.86, f1: 0.81, roc_auc: 0.69, average_precision: 0.79,
      brier_score: 0.2, confusion_matrix: { tn: 9, fp: 6, fn: 4, tp: 39 }, n_samples: 58, positive_rate: 0.74,
    },
    ensemble: {
      precision: 0.8, recall: 0.9, f1: 0.85, roc_auc: 0.71, average_precision: 0.82,
      brier_score: 0.18, confusion_matrix: { tn: 10, fp: 5, fn: 3, tp: 40 }, n_samples: 58, positive_rate: 0.74,
    },
  },
  best_hyperparameters: {
    logistic_regression: { C: 0.01 },
    random_forest: { n_estimators: 100, max_depth: 4 },
  },
  ensemble_members: ["logistic_regression", "random_forest"],
  disclaimer: "For planning support only.",
};

describe("ModelInfo page", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the model comparison table with every candidate, winner marked", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => MOCK_RESPONSE } as Response);

    render(<ModelInfo />);

    expect(await screen.findByText(/logistic regression/i)).toBeInTheDocument();
    expect(screen.getByText(/random forest/i)).toBeInTheDocument();
    // "ensemble" appears more than once (model-type stat card + comparison table row).
    expect(screen.getAllByText(/ensemble/i).length).toBeGreaterThanOrEqual(2);
    // The ensemble has the best (highest) ROC-AUC of the three in this fixture.
    expect(screen.getByText("0.710")).toBeInTheDocument();
  });

  it("renders the confusion matrix counts from test_metrics", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => MOCK_RESPONSE } as Response);

    render(<ModelInfo />);

    await screen.findByText(/confusion matrix/i);
    expect(screen.getByText("40")).toBeInTheDocument(); // true positives
    expect(screen.getByText("10")).toBeInTheDocument(); // true negatives
  });

  it("shows the calibration before/after Brier score comparison", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, json: async () => MOCK_RESPONSE } as Response);

    render(<ModelInfo />);

    expect(await screen.findByText(/0\.210.*0\.180/)).toBeInTheDocument();
  });

  it("shows an error message when the API call fails", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({ detail: "Model not loaded." }),
    } as Response);

    render(<ModelInfo />);

    expect(await screen.findByText("Model not loaded.")).toBeInTheDocument();
  });
});
