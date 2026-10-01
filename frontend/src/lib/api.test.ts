import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, getModelInfo, predict } from "./api";
import type { StudentTermFeatures } from "./types";

const VALID_RECORD: StudentTermFeatures = {
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

describe("api client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("predict() sends a POST with the feature payload and returns the parsed JSON", async () => {
    const mockResponse = { persistence_probability: 0.7, risk_category: "low" };
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const result = await predict(VALID_RECORD);

    expect(result).toEqual(mockResponse);
    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toContain("/predict");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(VALID_RECORD);
  });

  it("throws ApiError with the Pydantic field error joined into a readable message on 422", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        detail: [{ loc: ["body", "term_gpa"], msg: "Input should be less than or equal to 4" }],
      }),
    } as Response);

    await expect(predict(VALID_RECORD)).rejects.toThrow(ApiError);
    await expect(predict(VALID_RECORD)).rejects.toThrow(/term_gpa/);
  });

  it("throws ApiError with the plain string detail on other error statuses (e.g. 503)", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({ detail: "Model not loaded." }),
    } as Response);

    await expect(getModelInfo()).rejects.toThrow("Model not loaded.");
  });
});
