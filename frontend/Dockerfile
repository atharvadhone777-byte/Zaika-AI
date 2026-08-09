# Only used for containerized deployment of the frontend (e.g. if not
# using Vercel). Two stages: build the static bundle with Node, then serve
# it with a minimal nginx image - nginx doesn't need Node or the ~200MB of
# node_modules in the final image, only the ~250KB built bundle.

FROM node:20-slim AS builder
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# The API base URL is baked into the static bundle at BUILD time (Vite
# resolves import.meta.env.* at build time, not runtime) - pass it as a
# build arg so the same Dockerfile works against any backend URL without
# editing source.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:alpine
COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
