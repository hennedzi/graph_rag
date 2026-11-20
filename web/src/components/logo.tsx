export function Logo({ className = "h-7 w-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-label="Graph RAG">
      <rect width="32" height="32" rx="8" className="fill-primary/15" />
      <path d="M9 22 16 9l7 13z" fill="none" strokeWidth="2" className="stroke-primary" />
      <circle cx="16" cy="9" r="3" className="fill-primary" />
      <circle cx="9" cy="22" r="3" fill="hsl(var(--chart-2))" />
      <circle cx="23" cy="22" r="3" className="fill-primary/60" />
    </svg>
  );
}
