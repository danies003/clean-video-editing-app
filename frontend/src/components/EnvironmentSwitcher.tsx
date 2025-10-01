"use client";

import { useState, useEffect } from "react";
import { apiClient } from "@/lib/api";

export default function EnvironmentSwitcher() {
  const [currentBackend, setCurrentBackend] = useState<string>("");
  const [environment, setEnvironment] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [hasBackendOverride, setHasBackendOverride] = useState(false);
  const [isClient, setIsClient] = useState(false);

  // Environment detection
  const isDevelopment =
    process.env.NEXT_PUBLIC_ENVIRONMENT === "development" ||
    process.env.NODE_ENV === "development" ||
    (typeof window !== "undefined" && window.location.hostname === "localhost");

  useEffect(() => {
    setIsClient(true);
    loadEnvironmentInfo();
  }, []);

  useEffect(() => {
    if (isClient && typeof window !== "undefined") {
      const override = localStorage.getItem("backend_override");
      setHasBackendOverride(!!override);
    }
  }, [isClient]);

  const loadEnvironmentInfo = async () => {
    try {
      const backendUrl = await apiClient.getCurrentBackendURL();
      setCurrentBackend(backendUrl);
      setEnvironment(process.env.NEXT_PUBLIC_ENVIRONMENT || "unknown");
    } catch (error) {
      console.error("Failed to load environment info:", error);
    }
  };

  const handleRediscover = async () => {
    setIsLoading(true);
    try {
      await apiClient.clearCache();
      await loadEnvironmentInfo();
    } catch (error) {
      console.error("Failed to rediscover backend:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleForceLocalhost = async () => {
    setIsLoading(true);
    try {
      await apiClient.forceLocalhost();
      await loadEnvironmentInfo();
    } catch (error) {
      console.error("Failed to force localhost:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const clearOverride = () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem("backend_override");
      setHasBackendOverride(false);
      loadEnvironmentInfo();
    }
  };

  if (!isClient) {
    return null; // Don't render on server side
  }

  return (
    <div className="fixed bottom-4 right-4 bg-gray-800 text-white p-3 rounded-lg shadow-lg text-xs max-w-xs">
      <div className="font-semibold mb-2">Environment Status</div>

      <div className="space-y-1">
        <div>
          <span className="text-gray-300">Environment:</span>
          <span
            className={`ml-1 px-2 py-1 rounded text-xs ${
              environment === "development" ? "bg-green-600" : "bg-blue-600"
            }`}
          >
            {environment}
          </span>
        </div>

        <div>
          <span className="text-gray-300">Backend:</span>
          <div className="text-xs break-all mt-1">{currentBackend}</div>
        </div>
      </div>

      <div className="mt-3 space-y-2">
        <button
          onClick={handleRediscover}
          disabled={isLoading}
          className="w-full px-2 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded text-xs"
        >
          {isLoading ? "Rediscovering..." : "Rediscover Backend"}
        </button>

        {isDevelopment && (
          <button
            onClick={handleForceLocalhost}
            disabled={isLoading}
            className="w-full px-2 py-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded text-xs"
          >
            {isLoading ? "Forcing..." : "Force Localhost"}
          </button>
        )}

        {hasBackendOverride && (
          <button
            onClick={clearOverride}
            className="w-full px-2 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-xs"
          >
            Clear Override
          </button>
        )}
      </div>
    </div>
  );
}
