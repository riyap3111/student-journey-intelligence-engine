// Default for local dev (npm run dev) and the production build before any
// runtime substitution: no override, so api.ts falls back to
// VITE_API_BASE_URL / its own default. In a container, docker-entrypoint.sh
// overwrites this file from the API_BASE_URL env var before nginx starts.
window.__ENV__ = {};
