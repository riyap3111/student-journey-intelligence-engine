# Frontend — Student Journey Intelligence Engine

A React + TypeScript + Vite + Tailwind single-page app that talks to the FastAPI backend
(`../src/student_journey/api/`) over HTTP: student-level predictions, model info, and live
PSI drift monitoring, with a UI meant to look like a real product rather than a
data-science dashboard. The Streamlit dashboard (`../src/student_journey/dashboard/`)
still exists separately and is unaffected by this — they're two different front ends over
the same backend.

## Run locally

In one terminal, the backend (see the repo root README for full setup):
```bash
cd ..
source .venv/bin/activate
export PYTHONPATH=src
uvicorn student_journey.api.main:app --port 8000
```

In another terminal, this app:
```bash
npm install
npm run dev
# opens at http://localhost:5173
```

The backend's CORS config (`CORS_ALLOWED_ORIGINS` in `main.py`) already allows
`http://localhost:5173` by default, so no extra setup is needed for local dev.

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Type-check (`tsc -b`) + production build to `dist/` |
| `npm run lint` | oxlint |
| `npm run test` | Vitest — component tests (`RiskBadge`), the API client's error parsing, and a full Predict-form submission flow against a mocked `fetch` |
| `npm run preview` | Serve the production build locally |

## How the API URL is configured

Vite normally bakes environment variables into the JS bundle at *build* time
(`VITE_API_BASE_URL`, see `.env.example`) — fine for local dev, but a problem for Cloud
Run: the frontend image would need rebuilding every time the API's URL changed, and the
API's URL isn't even known until after it's deployed.

Instead, `index.html` loads `/env.js` before the app bundle, and `src/lib/api.ts` checks
`window.__ENV__.API_BASE_URL` first, falling back to the build-time `VITE_API_BASE_URL`.
In Docker, `docker-entrypoint.sh` regenerates `env.js` from the container's
`API_BASE_URL` env var at **startup**, not build time — so one built image can be
deployed against any backend. See `Dockerfile` and the repo root's
`docs/deployment/gcp.md`.

## Not yet done

- No end-to-end (browser) tests — `npm run test` covers components and the API client
  with a mocked `fetch`, not a real running backend. The Predict flow *was* verified
  manually against a real local API during development (see the repo root README's
  frontend section) — just not as an automated test.
- No code-splitting — a single JS bundle (~190KB gzipped, mostly Recharts + React
  Router); fine at this size, called out by Vite's own build warning as a future
  optimization if the app grows.
