"use client";

import { useState, useEffect } from "react";
import { apiClient } from "@/lib/api";

interface BackendStatusProps {
  className?: string;
}

export default function BackendStatus({ className = "" }: BackendStatusProps) {
  const [status, setStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");
  const [backendUrl, setBackendUrl] = useState<string>("");
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  useEffect(() => {
    checkBackendStatus();
  }, []);

  const checkBackendStatus = async () => {
    setStatus("checking");
    console.log("🔍 Starting backend status check...");

    // Add a small delay to ensure backend is fully ready
    await new Promise((resolve) => setTimeout(resolve, 1000));
    console.log("🔍 Delay completed, proceeding with check...");

    try {
      console.log("🔍 Getting current backend URL...");
      const url = await apiClient.getCurrentBackendURL();
      console.log("✅ Backend URL obtained:", url);
      setBackendUrl(url);

      console.log("🔍 Performing health check...");
      await apiClient.healthCheck();
      console.log("✅ Health check successful");
      setStatus("connected");
      setLastCheck(new Date());
    } catch (error) {
      console.error("❌ Backend check failed:", error);
      console.error("❌ Error details:", {
        name: error instanceof Error ? error.name : "Unknown",
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : "No stack trace",
      });
      setStatus("disconnected");
      setLastCheck(new Date());
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case "connected":
        return "text-green-600";
      case "disconnected":
        return "text-red-600";
      case "checking":
        return "text-yellow-600";
      default:
        return "text-gray-600";
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case "connected":
        return "🟢";
      case "disconnected":
        return "🔴";
      case "checking":
        return "🟡";
      default:
        return "⚪";
    }
  };

  return (
    <div className={`text-sm ${className}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={getStatusColor()}>
          {getStatusIcon()} Backend: {status}
        </span>
        <button
          onClick={checkBackendStatus}
          className="text-blue-600 hover:text-blue-800 text-xs underline"
        >
          Refresh
        </button>
      </div>

      {backendUrl && (
        <div className="text-xs text-gray-600 mb-2">URL: {backendUrl}</div>
      )}

      {lastCheck && (
        <div className="text-xs text-gray-500 mb-2">
          Last check: {lastCheck.toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}
