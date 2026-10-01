/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

interface Window {
  // Written by docker-entrypoint.sh at container startup (see Dockerfile);
  // absent in local `npm run dev`, where api.ts falls back to VITE_API_BASE_URL.
  __ENV__?: { API_BASE_URL?: string };
}
