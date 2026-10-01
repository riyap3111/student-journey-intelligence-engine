import { useState } from "react";

export default function CodeBlock({ code, label }: { code: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard access can be denied (e.g. insecure context); the code is still visible to copy manually */
    }
  };

  return (
    <div className="relative overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-black/30">
      <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-3 py-1.5">
        <span className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{label}</span>
        <button
          type="button"
          onClick={copy}
          className="rounded px-2 py-0.5 text-xs text-[var(--text-secondary)] transition-colors hover:bg-white/5 hover:text-[var(--text-primary)]"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="font-mono-ui overflow-x-auto p-3 text-xs leading-relaxed text-[var(--text-secondary)]">{code}</pre>
    </div>
  );
}
