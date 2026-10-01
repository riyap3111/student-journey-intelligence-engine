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
    <div className="mx-auto max-w-5xl space-y-8 px-4 py-14">
      <div>
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[var(--border-subtle)] bg-white/[0.02] px-3 py-1 text-xs font-medium text-[var(--text-secondary)]">
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent-bright)]" />
          Portfolio ML system · synthetic data
        </div>
        <h1 className="font-display text-4xl font-bold leading-tight text-[var(--text-primary)] sm:text-5xl">
          Predicting student{" "}
          <span className="bg-gradient-to-r from-[var(--accent-bright)] to-[var(--cyan)] bg-clip-text text-transparent">
            persistence
          </span>
          , explained.
        </h1>
        <p className="mt-4 max-w-2xl text-[var(--text-secondary)]">
          An ensemble model estimates whether a student continues enrollment next term, explains the
          factors behind every prediction with SHAP, and ships with live feature-drift monitoring — served
          by a FastAPI backend and this React frontend.
        </p>
      </div>

      <PrivacyNotice />

      {error && (
        <div className="glass-panel border-[var(--risk-high)]/30 p-4 text-sm text-[var(--text-secondary)]">
          Can't reach the API at{" "}
          <code className="font-mono-ui rounded bg-white/5 px-1.5 py-0.5 text-[var(--risk-high)]">
            {import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}
          </code>
          . Start it with:
          <pre className="font-mono-ui mt-2 overflow-x-auto rounded-lg border border-[var(--border-subtle)] bg-black/30 p-3 text-xs text-[var(--text-secondary)]">
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
            accent="cyan"
          />
          {modelInfo && (
            <>
              <StatCard label="Test ROC-AUC" value={Number(modelInfo.test_metrics.roc_auc).toFixed(3)} />
              <StatCard
                label="Model type"
                value={modelInfo.model_type.replace("_", " ")}
                accent="cyan"
              />
            </>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-3 pt-2">
        <Link
          to="/predict"
          className="rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-medium text-white shadow-[0_0_24px_-4px_var(--accent-glow)] transition-transform hover:scale-[1.02] hover:bg-[var(--accent-bright)]"
        >
          Try a prediction →
        </Link>
        <Link
          to="/monitoring"
          className="rounded-full border border-[var(--border-strong)] bg-white/[0.02] px-5 py-2.5 text-sm font-medium text-[var(--text-primary)] transition-colors hover:bg-white/5"
        >
          View drift monitoring
        </Link>
        <a
          href="https://github.com/riyap3111/student-journey-intelligence-engine"
          target="_blank"
          rel="noreferrer"
          className="rounded-full border border-[var(--border-strong)] bg-white/[0.02] px-5 py-2.5 text-sm font-medium text-[var(--text-primary)] transition-colors hover:bg-white/5"
        >
          View source on GitHub
        </a>
      </div>
    </div>
  );
}
