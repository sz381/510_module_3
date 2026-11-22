import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const api = (process.env.MODEL_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
    return [
      { source: "/api/metadata", destination: api + "/metadata" },
      { source: "/api/predict", destination: api + "/predict" },
    ];
  },
  experimental: {
    agentFeedback: true,
  },
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
