# Multi-stage build: compile the Vite app, then serve the static output with
# a minimal nginx image — no Node.js runtime needed in the final image.
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Single-page app: unknown paths (e.g. a refresh on /predict) must fall back
# to index.html so React Router can handle the route client-side.
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    location / {\n\
        try_files $uri /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
EXPOSE 80
ENTRYPOINT ["/docker-entrypoint.sh"]
