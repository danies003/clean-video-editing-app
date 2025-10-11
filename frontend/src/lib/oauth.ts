// OAuth Configuration and Utilities

export interface OAuthConfig {
  google: {
    enabled: boolean;
    client_id?: string;
  };
  facebook: {
    enabled: boolean;
    app_id?: string;
  };
}

export interface OAuthUser {
  id: string;
  email: string;
  name: string;
  picture?: string;
  provider: "google" | "facebook" | "email";
}

// Environment-aware configuration
const isDevelopment = process.env.NODE_ENV === "development";
const baseUrl = isDevelopment
  ? "http://localhost:3000"
  : "https://your-app.railway.app";

// Google OAuth
export const initGoogleOAuth = async (): Promise<boolean> => {
  try {
    // Load Google OAuth script
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;

    return new Promise((resolve) => {
      script.onload = () => resolve(true);
      script.onerror = () => resolve(false);
      document.head.appendChild(script);
    });
  } catch (error) {
    console.error("Failed to load Google OAuth script:", error);
    return false;
  }
};

export const signInWithGoogle = async (): Promise<OAuthUser | null> => {
  try {
    console.log("Starting Google OAuth flow...");

    // Get OAuth config from backend
    const configResponse = await fetch(`/api/v1/auth/config`);
    const config: OAuthConfig = await configResponse.json();
    console.log("OAuth config received:", config);

    if (!config.google.enabled || !config.google.client_id) {
      throw new Error("Google OAuth not configured");
    }

    // Create the Google OAuth URL
    const googleOAuthUrl = new URL(
      "https://accounts.google.com/o/oauth2/v2/auth"
    );
    googleOAuthUrl.searchParams.set("client_id", config.google.client_id);
    googleOAuthUrl.searchParams.set(
      "redirect_uri",
      "http://localhost:3000/auth/callback"
    );
    googleOAuthUrl.searchParams.set("response_type", "code");
    googleOAuthUrl.searchParams.set("scope", "openid email profile");
    googleOAuthUrl.searchParams.set("access_type", "offline");
    googleOAuthUrl.searchParams.set("prompt", "select_account");

    console.log("Redirecting to Google OAuth:", googleOAuthUrl.toString());

    // Store the current page URL to return to after OAuth
    sessionStorage.setItem("oauth_return_url", window.location.href);

    // Redirect to Google OAuth
    window.location.href = googleOAuthUrl.toString();

    // This will never be reached due to redirect, but TypeScript needs it
    return null;
  } catch (error) {
    console.error("Google OAuth error:", error);
    throw error;
  }
};

// Facebook OAuth
export const initFacebookOAuth = async (): Promise<boolean> => {
  try {
    // Load Facebook SDK
    const script = document.createElement("script");
    script.src = "https://connect.facebook.net/en_US/sdk.js";
    script.async = true;
    script.defer = true;

    return new Promise((resolve) => {
      script.onload = () => resolve(true);
      script.onerror = () => resolve(false);
      document.head.appendChild(script);
    });
  } catch (error) {
    console.error("Failed to load Facebook SDK:", error);
    return false;
  }
};

export const signInWithFacebook = async (): Promise<OAuthUser | null> => {
  // Not used in redirect flow. Keep for compatibility if needed.
  return null;
};

// Facebook OAuth (manual token path for development/debug)
export const signInWithFacebookManual = async (
  accessToken: string
): Promise<OAuthUser | null> => {
  try {
    // Use the same API URL as the main API client
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    const resp = await fetch(`${API_URL}/api/v1/auth/social`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: "facebook",
        token: accessToken,
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || "Authentication failed");
    }

    const authData = await resp.json();
    return {
      id: authData.user.id,
      email: authData.user.email,
      name: authData.user.name,
      provider: "facebook",
    };
  } catch (error) {
    console.error("Facebook manual token sign-in failed:", error);
    return null;
  }
};

// Email authentication
export const signInWithEmail = async (
  email: string,
  password: string,
  isSignUp: boolean = false,
  name?: string
): Promise<OAuthUser | null> => {
  try {
    const endpoint = isSignUp ? "/signup" : "/signin";
    const payload = isSignUp
      ? { name, email, password, provider: "email" }
      : { email, password };

    const response = await fetch(`/api/v1/auth${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error("Authentication failed");
    }

    const authData = await response.json();
    return {
      id: authData.user.id,
      email: authData.user.email,
      name: authData.user.name,
      provider: "email",
    };
  } catch (error) {
    console.error("Email authentication failed:", error);
    return null;
  }
};
