"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";

function AuthCallbackContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [status, setStatus] = useState<"loading" | "success" | "error">(
    "loading"
  );
  const [message, setMessage] = useState("Completing sign-in...");

  useEffect(() => {
    const handleOAuthCallback = async () => {
      const code = searchParams.get("code");
      const error = searchParams.get("error");
      const state = searchParams.get("state");

      if (error) {
        setStatus("error");
        setMessage("Authentication failed. Please try again.");
        setTimeout(() => {
          router.push("/");
        }, 3000);
        return;
      }

      if (code) {
        // Determine provider from state or session storage; default to google
        const storedProvider =
          typeof window !== "undefined"
            ? sessionStorage.getItem("oauth_provider")
            : null;
        const provider = (state || storedProvider || "google").toLowerCase();

        try {
          // Use HTTPS for OAuth completion (required by Facebook)
          const response = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/auth/social`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                provider: provider,
                code: code,
                redirect_uri: `${window.location.origin}/auth/callback`, // Must match the redirect_uri used to obtain the code
              }),
            }
          );

          if (!response.ok) {
            throw new Error("Authentication failed");
          }
          const authData = await response.json();

          setStatus("success");
          setMessage("Authentication successful! Redirecting...");

          // Store the auth data
          localStorage.setItem("auth_token", authData.access_token);
          localStorage.setItem("user", JSON.stringify(authData.user));

          console.log("✅ Authentication data stored:", {
            token: authData.access_token ? "present" : "missing",
            user: authData.user,
          });

          // Redirect back to the original page or home
          const returnUrl = sessionStorage.getItem("oauth_return_url") || "/";
          sessionStorage.removeItem("oauth_provider");
          sessionStorage.removeItem("oauth_return_url");

          console.log("🔄 Redirecting to:", returnUrl);

          setTimeout(() => {
            router.push(returnUrl);
          }, 2000); // Increased delay to ensure state is set
        } catch (error) {
          console.error("Authentication error:", error);
          setStatus("error");
          setMessage("Authentication failed. Please try again.");
          setTimeout(() => {
            router.push("/");
          }, 3000);
        }
      } else {
        setStatus("error");
        setMessage("No authorization code received. Please try again.");
        setTimeout(() => {
          router.push("/");
        }, 3000);
      }
    };

    handleOAuthCallback();
  }, [searchParams, router]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <div className="text-center">
        {status === "loading" && (
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        )}
        {status === "success" && (
          <div className="text-green-600 mb-4">
            <svg
              className="w-12 h-12 mx-auto"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
        )}
        {status === "error" && (
          <div className="text-red-600 mb-4">
            <svg
              className="w-12 h-12 mx-auto"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </div>
        )}
        <p className="text-gray-600">{message}</p>
      </div>
    </div>
  );
}

export default function AuthCallback() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-screen bg-gray-50">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading authentication...</p>
          </div>
        </div>
      }
    >
      <AuthCallbackContent />
    </Suspense>
  );
}
