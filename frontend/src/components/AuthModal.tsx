"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, ArrowLeft, Mail, Lock, Eye, EyeOff } from "lucide-react";
import {
  signInWithGoogle,
  signInWithFacebook,
  signInWithFacebookManual,
  signInWithEmail,
  OAuthUser,
} from "@/lib/oauth";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAuthSuccess: (user: OAuthUser) => void;
}

export default function AuthModal({
  isOpen,
  onClose,
  onAuthSuccess,
}: AuthModalProps) {
  const [isSignUp, setIsSignUp] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  // Independent loading flags per action to avoid locking the whole modal
  const [googleLoading, setGoogleLoading] = useState(false);
  const [facebookLoading, setFacebookLoading] = useState(false);
  const [facebookManualLoading, setFacebookManualLoading] = useState(false);
  const [emailLoading, setEmailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
  });
  const [fbToken, setFbToken] = useState("");

  // Debug logging
  console.log("AuthModal render:", {
    isOpen,
    shouldShow: isOpen,
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setEmailLoading(true);
    setError(null);

    try {
      const user = await signInWithEmail(
        formData.email,
        formData.password,
        isSignUp,
        formData.name
      );
      if (user) {
        onAuthSuccess(user);
      }
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Authentication failed"
      );
    } finally {
      setEmailLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setGoogleLoading(true);
    setError(null);
    try {
      console.log("Starting Google sign-in...");
      const user = await signInWithGoogle();
      if (user) {
        console.log("Google sign-in successful:", user);
        onAuthSuccess(user);
      } else {
        console.error("Google sign-in returned null user");
        setError("Google sign-in failed. Please try again.");
      }
    } catch (error) {
      console.error("Google sign-in failed:", error);
      setError(
        `Google sign-in failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      setGoogleLoading(false);
    }
  };

  const handleFacebookSignIn = async () => {
    setFacebookLoading(true);
    setError(null);
    try {
      // Use dynamic API URL from environment instead of hardcoded localhost:8443
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const configResp = await fetch(`${apiUrl}/api/v1/auth/config`, {
        mode: "cors",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const config = await configResp.json();
      const appId = config?.facebook?.app_id;
      if (!appId) throw new Error("Facebook app id not configured");

      // Store provider info for callback
      sessionStorage.setItem("oauth_provider", "facebook");
      sessionStorage.setItem("oauth_return_url", window.location.pathname);

      const redirectUri = encodeURIComponent(
        `${window.location.origin}/auth/callback`
      );
      const facebookAuthUrl = `https://www.facebook.com/v19.0/dialog/oauth?client_id=${appId}&redirect_uri=${redirectUri}&response_type=code&scope=email,public_profile&state=facebook`;
      window.location.href = facebookAuthUrl;
    } catch (error) {
      console.error("Error initiating Facebook sign-in:", error);
      setError(
        `Facebook sign-in failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      setFacebookLoading(false);
    }
  };

  const handleFacebookManual = async () => {
    setFacebookManualLoading(true);
    setError(null);
    try {
      if (!fbToken) {
        setError("Paste a Facebook user access token");
        return;
      }
      const user = await signInWithFacebookManual(fbToken);
      if (user) {
        onAuthSuccess(user);
      } else {
        setError("Facebook manual token sign-in failed.");
      }
    } catch (error) {
      setError(
        `Facebook manual sign-in failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      setFacebookManualLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          style={{ opacity: 1, visibility: "visible" }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="w-full max-w-md"
            style={{ opacity: 1, transform: "scale(1)" }}
          >
            <div className="rounded-lg border text-card-foreground bg-white shadow-2xl">
              <div className="flex flex-col space-y-1.5 p-6 relative">
                <button
                  onClick={onClose}
                  className="absolute top-4 right-4 p-2 hover:bg-gray-100 rounded-full transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
                <div className="flex items-center gap-3 mb-2">
                  <button
                    onClick={() => setIsSignUp(!isSignUp)}
                    className="p-1 hover:bg-gray-100 rounded-full transition-colors"
                  >
                    <ArrowLeft className="w-4 h-4" />
                  </button>
                  <h3 className="tracking-tight text-xl font-semibold">
                    {isSignUp ? "Sign up" : "Sign in"}
                  </h3>
                </div>
                <p className="text-sm text-gray-600">
                  {isSignUp
                    ? "Create your account"
                    : "Welcome back! Sign in to continue"}
                </p>
              </div>

              <div className="p-6 pt-0 space-y-4">
                <div className="space-y-3">
                  <button
                    onClick={handleGoogleSignIn}
                    disabled={googleLoading}
                    className="justify-center rounded-xl text-sm transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border-2 backdrop-blur-md hover:border-gray-300 px-8 py-2 w-full h-12 flex items-center gap-3 bg-white hover:bg-gray-50 border-gray-300 text-gray-700 font-medium"
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24">
                      <path
                        fill="#4285F4"
                        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                      />
                      <path
                        fill="#34A853"
                        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                      />
                      <path
                        fill="#FBBC05"
                        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                      />
                      <path
                        fill="#EA4335"
                        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                      />
                    </svg>
                    {googleLoading ? "Signing in..." : "Continue with Google"}
                  </button>

                  <button
                    onClick={handleFacebookSignIn}
                    disabled={facebookLoading}
                    className="justify-center rounded-xl text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border-2 backdrop-blur-md hover:border-gray-300 px-8 py-2 w-full h-12 flex items-center gap-3 bg-blue-600 hover:bg-blue-700 text-white border-blue-600"
                  >
                    <svg
                      className="w-5 h-5"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
                    </svg>
                    Continue with Facebook
                  </button>

                  {/* Dev-only manual token sign-in */}
                  <div className="pt-2 space-y-2">
                    <input
                      type="text"
                      value={fbToken}
                      onChange={(e) => setFbToken(e.target.value)}
                      placeholder="Paste Facebook user access token"
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    />
                    <button
                      onClick={handleFacebookManual}
                      disabled={facebookManualLoading}
                      className="justify-center rounded-xl text-sm transition-all duration-200 border px-8 py-2 w-full h-10 bg-white hover:bg-gray-50 border-gray-300 text-gray-700"
                    >
                      {facebookManualLoading
                        ? "Verifying..."
                        : "Use Token (Dev)"}
                    </button>
                  </div>
                </div>

                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-white px-2 text-gray-500">
                      Or continue with email
                    </span>
                  </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                  {isSignUp && (
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700">
                        Name *
                      </label>
                      <div className="relative">
                        <input
                          type="text"
                          name="name"
                          value={formData.name}
                          onChange={handleInputChange}
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                          placeholder="Enter your name"
                          required={isSignUp}
                        />
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">
                      Email *
                    </label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                      <input
                        type="email"
                        name="email"
                        value={formData.email}
                        onChange={handleInputChange}
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 pl-10"
                        placeholder="Enter your email"
                        required
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">
                      Password *
                    </label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                      <input
                        type={showPassword ? "text" : "password"}
                        name="password"
                        value={formData.password}
                        onChange={handleInputChange}
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 pl-10 pr-10"
                        placeholder="Enter your password"
                        required
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                      >
                        {showPassword ? (
                          <EyeOff className="w-4 h-4" />
                        ) : (
                          <Eye className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={emailLoading}
                    className="inline-flex items-center justify-center rounded-xl text-sm transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 backdrop-blur-sm shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] px-8 py-2 w-full h-12 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-medium"
                  >
                    {emailLoading
                      ? "Loading..."
                      : isSignUp
                      ? "Sign up"
                      : "Sign in"}
                  </button>
                </form>

                {error && (
                  <div className="text-center text-red-600 text-sm">
                    {error}
                  </div>
                )}

                <div className="text-center">
                  <p className="text-sm text-gray-600">
                    {isSignUp
                      ? "Already have an account? "
                      : "Don't have an account? "}
                    <button
                      onClick={() => setIsSignUp(!isSignUp)}
                      className="text-purple-600 hover:underline font-medium"
                    >
                      {isSignUp ? "Sign in" : "Sign up"}
                    </button>
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
