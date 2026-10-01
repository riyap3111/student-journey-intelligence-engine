import type {
  ApiErrorBody,
  DriftReportResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
  StudentTermFeatures,
} from "./types";

// Resolution order: a runtime-injected value (window.__ENV__, written by
// docker-entrypoint.sh from the API_BASE_URL container env var — see
// Dockerfile) takes priority over the build-time VITE_API_BASE_URL (see
// .env.example, used for local `npm run dev`), which falls back to the
// local FastAPI dev server. Runtime injection matters because Vite env
// vars are baked into the JS bundle at build time, but Cloud Run doesn't
// know the API service's URL until *after* deploying it — a static image
// built once and configured per-deployment via an env var (exactly how
// this project's API/dashboard image already picks its role) avoids
// rebuilding the frontend image just to point it at a different backend.
export const API_BASE_URL: string =
  window.__ENV__?.API_BASE_URL ?? import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ApiErrorBody | null;
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => `${d.loc.join(".")}: ${d.msg}`).join("; ")
          : `Request failed with status ${response.status}`;
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

export const getHealth = () => request<HealthResponse>("/health");

export const getModelInfo = () => request<ModelInfoResponse>("/model_info");

export const getDriftReport = () => request<DriftReportResponse>("/monitoring/drift");

export const predict = (features: StudentTermFeatures) =>
  request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(features),
  });
