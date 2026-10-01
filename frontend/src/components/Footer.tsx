import { API_BASE_URL } from "../lib/api";

const STACK = [
  "Python", "FastAPI", "scikit-learn", "XGBoost", "Optuna", "SHAP", "MLflow",
  "SQLAlchemy", "React", "TypeScript", "Vite",
];

const LINKS = [
  { href: "https://github.com/riyap3111/student-journey-intelligence-engine", label: "Source" },
  { href: `${API_BASE_URL}/docs`, label: "API reference (Swagger)" },
  {
    href: "https://github.com/riyap3111/student-journey-intelligence-engine/blob/main/docs/model_card.md",
    label: "Model card",
  },
  {
    href: "https://github.com/riyap3111/student-journey-intelligence-engine/blob/main/docs/architecture.md",
    label: "Architecture",
  },
  {
    href: "https://github.com/riyap3111/student-journey-intelligence-engine/actions",
    label: "CI runs",
  },
];

export default function Footer() {
  return (
    <footer className="mt-16 border-t border-[var(--border-subtle)]">
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="flex flex-wrap gap-1.5">
          {STACK.map((tech) => (
            <span
              key={tech}
              className="font-mono-ui rounded-md border border-[var(--border-subtle)] bg-white/[0.02] px-2 py-1 text-[10px] text-[var(--text-muted)]"
            >
              {tech}
            </span>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-xs">
          {LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noreferrer"
              className="text-[var(--text-secondary)] underline decoration-[var(--border-strong)] underline-offset-4 transition-colors hover:text-[var(--accent-bright)] hover:decoration-[var(--accent-bright)]"
            >
              {link.label}
            </a>
          ))}
        </div>
        <p className="mt-4 text-xs text-[var(--text-muted)]">
          Portfolio project using entirely synthetic data. Not affiliated with any real university. 84 Python
          tests + 9 frontend tests, CI on every push.
        </p>
      </div>
    </footer>
  );
}
