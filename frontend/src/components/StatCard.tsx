export default function StatCard({
  label,
  value,
  caption,
  accent = "violet",
}: {
  label: string;
  value: string;
  caption?: string;
  accent?: "violet" | "cyan";
}) {
  const accentColor = accent === "cyan" ? "#22d3ee" : "#a78bfa";
  return (
    <div className="glass-panel relative overflow-hidden p-4">
      <div
        className="absolute inset-x-0 top-0 h-px"
        style={{ background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)` }}
      />
      <div className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
      <div className="mt-1.5 font-display text-2xl font-semibold text-[var(--text-primary)]">{value}</div>
      {caption && <div className="mt-1 text-xs text-[var(--text-secondary)]">{caption}</div>}
    </div>
  );
}
