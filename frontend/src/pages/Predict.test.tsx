import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Predict from "./Predict";

describe("Predict page", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("submits the form and renders the returned prediction", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        persistence_probability: 0.65,
        risk_probability: 0.35,
        risk_category: "medium",
        score_scale: "probability",
        top_contributing_factors: [
          { feature: "term_gpa", contribution: -0.09, direction: "increases_risk" },
        ],
        model_version: "test-version",
        disclaimer: "Planning and advising support only.",
      }),
    } as Response);

    render(<Predict />);
    await userEvent.click(screen.getByRole("button", { name: /predict/i }));

    await waitFor(() => expect(screen.getByText("MEDIUM RISK")).toBeInTheDocument());
    expect(screen.getByText(/65\.0%/)).toBeInTheDocument();
    expect(screen.getByText(/test-version/)).toBeInTheDocument();
  });

  it("shows an error message when the API call fails", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({ detail: "Model not loaded." }),
    } as Response);

    render(<Predict />);
    await userEvent.click(screen.getByRole("button", { name: /predict/i }));

    await waitFor(() => expect(screen.getByText("Model not loaded.")).toBeInTheDocument());
  });

  it("reflects a changed field in the submitted request body", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        persistence_probability: 0.5,
        risk_probability: 0.5,
        risk_category: "high",
        score_scale: "probability",
        top_contributing_factors: [],
        model_version: "test-version",
        disclaimer: "",
      }),
    } as Response);

    render(<Predict />);
    const termNumberInput = screen.getByLabelText(/term number/i);
    await userEvent.clear(termNumberInput);
    await userEvent.type(termNumberInput, "7");
    await userEvent.click(screen.getByRole("button", { name: /predict/i }));

    await waitFor(() => expect(fetch).toHaveBeenCalled());
    const [, init] = vi.mocked(fetch).mock.calls[0];
    const body = JSON.parse(init?.body as string);
    expect(body.term_number).toBe(7);
  });
});
