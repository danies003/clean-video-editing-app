import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // output: "standalone", // Commented out for Railway deployment
  env: {
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL ||
      "https://backend-production-acb4.up.railway.app",
  },
  experimental: {
    serverActions: {
      bodySizeLimit: "50mb",
    },
  },
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "@": path.resolve(__dirname, "src"),
    };
    return config;
  },
  async rewrites() {
    // Only use rewrites in development
    if (process.env.NODE_ENV === "development") {
      return [
        {
          source: "/api/:path*",
          destination: "http://localhost:8000/api/v1/:path*",
        },
      ];
    }
    return [];
  },
};

export default nextConfig;
