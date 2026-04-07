import type { NextConfig } from "next";

const backendBaseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendBaseUrl}/api/:path*`,
      },
      {
        source: "/chat",
        destination: `${backendBaseUrl}/chat`,
      },
      {
        source: "/chat/:path*",
        destination: `${backendBaseUrl}/chat/:path*`,
      },
      {
        source: "/tts",
        destination: `${backendBaseUrl}/tts`,
      },
      {
        source: "/health",
        destination: `${backendBaseUrl}/health`,
      },
      {
        source: "/scrape",
        destination: `${backendBaseUrl}/scrape`,
      },
    ];
  },
};

export default nextConfig;
