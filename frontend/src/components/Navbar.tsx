import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Overview", end: true },
  { to: "/predict", label: "Predict" },
  { to: "/model", label: "Model Info" },
  { to: "/monitoring", label: "Monitoring" },
];

export default function Navbar() {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
        <div className="font-semibold text-slate-900">Student Journey Intelligence Engine</div>
        <nav className="flex gap-1">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
