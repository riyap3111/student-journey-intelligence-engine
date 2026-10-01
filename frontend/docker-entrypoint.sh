#!/bin/sh
# Writes the API base URL into env.js from the API_BASE_URL container env var
# BEFORE nginx starts serving — this is what lets one built image be pointed
# at any backend (local Compose, a deployed Cloud Run API) without rebuilding
# it, since Vite otherwise bakes env vars into the JS bundle at build time.
set -e

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
cat > /usr/share/nginx/html/env.js <<EOF
window.__ENV__ = { API_BASE_URL: "${API_BASE_URL}" };
EOF

exec nginx -g 'daemon off;'
