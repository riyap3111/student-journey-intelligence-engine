export default function PrivacyNotice() {
  return (
    <div className="glass-panel flex gap-3 border-[var(--cyan)]/20 p-4 text-sm text-[var(--text-secondary)]">
      <span className="mt-0.5 shrink-0 text-[var(--cyan)]">●</span>
      <p>
        <strong className="text-[var(--text-primary)]">Synthetic data only.</strong> Every student record
        behind this app is fabricated by a stochastic simulation — no real student, institution, or record
        is represented. This is a portfolio/educational project and is not affiliated with, endorsed by, or
        in use at any real university. Predictions are for planning and advising support only, never an
        automated decision about any student.
      </p>
    </div>
  );
}
