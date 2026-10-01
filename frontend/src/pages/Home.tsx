import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getHealth, getModelInfo } from "../lib/api";
import PrivacyNotice from "../components/PrivacyNotice";
import StatCard from "../components/StatCard";
import type { HealthResponse, ModelInfoResponse } from "../lib/types";

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setError("unreachable"));
    getModelInfo()
      .then(setModelInfo)
      .catch(() => {
        /* model_info 503s until a model is trained; health check above already surfaces connectivity issues */
      });
  }, []);

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Student Journey Intelligence Engine</h1>
        <p className="mt-2 max-w-3xl text-slate-600">
          Predicts whether a student is likely to continue enrollment next term, explains the main
          contributing factors, and surfaces live feature-drift monitoring — served by a FastAPI
          backend and this React frontend.
        </p>
      </div>

      <PrivacyNotice />

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          Can't reach the API at <code className="rounded bg-red-100 px-1">{import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}</code>.
          Start it with:
          <pre className="mt-2 overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
            export PYTHONPATH=src{"\n"}uvicorn student_journey.api.main:app --port 8000
          </pre>
        </div>
      )}

      {health && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard label="API status" value={health.status === "ok" ? "Online" : "Down"} />
          <StatCard
            label="Model loaded"
            value={health.model_loaded ? "Yes" : "No"}
            caption={health.model_version ?? undefined}
          />
          {modelInfo && (
            <>
              <StatCard
                label="Test ROC-AUC"
                value={Number(modelInfo.test_metrics.roc_auc).toFixed(3)}
              />
              <StatCard label="Model type" value={modelInfo.model_type.replace("_", " ")} />
            </>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-3 pt-2">
        <Link
          to="/predict"
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
        >
          Try a prediction →
        </Link>
        <Link
          to="/monitoring"
          className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
        >
          View drift monitoring
        </Link>
        <a
          href="https://github.com/riyap3111/student-journey-intelligence-engine"
          target="_blank"
          rel="noreferrer"
          className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
        >
          View source on GitHub
        </a>
      </div>
    </div>
  );
}
