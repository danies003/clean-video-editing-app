"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Sparkles, Video, Star, Zap, Github } from "lucide-react";
import AdvancedUploadPanel from "@/components/AdvancedUploadPanel";
import PreviewPage from "@/components/PreviewPage";
import { VideoEditor } from "@/components/VideoEditor";
import { EnhancedVideoEditor } from "@/components/EnhancedVideoEditor";
import BackendStatus from "@/components/BackendStatus";
import EnvironmentSwitcher from "@/components/EnvironmentSwitcher";
import { multiVideoApi } from "@/lib/api";
import MultiVideoPreviewPage from "@/components/MultiVideoPreviewPage";
import AuthModal from "@/components/AuthModal";
import { OAuthUser } from "@/lib/oauth";

type EditStyle = "tiktok" | "youtube" | "cinematic";
type TemplateType =
  | "beat_match"
  | "cinematic"
  | "fast_paced"
  | "slow_motion"
  | "transition_heavy"
  | "minimal";
type QualityPreset = "low" | "medium" | "high" | "ultra";

interface UploadParams {
  file: File;
  editStyle: EditStyle;
  templateType: TemplateType;
  qualityPreset: QualityPreset;
  skipPreview: boolean;
}

interface MultiUploadParams {
  files: File[];
  editStyle: EditStyle;
  templateType: TemplateType;
  qualityPreset: QualityPreset;
  crossVideoSettings: {
    enableCrossAnalysis: boolean;
    similarityThreshold: number;
    chunkingStrategy: "scene" | "action" | "audio" | "content";
  };
}

interface EditDecisionSegment {
  start: number;
  end: number;
  transition?: string;
  transition_duration?: number;
  tags: string[];
  speed?: number;
  transition_in?: string;
  transition_out?: string;
}

interface EditDecisionMap {
  video_id: string;
  style: string;
  segments: EditDecisionSegment[];
  notes?: string;
  edit_scale: number;
}

interface LLMSuggestion {
  type: "effect" | "transition" | "timing" | "style";
  title: string;
  description: string;
  reasoning: string;
  confidence: number;
  applied: boolean;
  segment_index?: number;
  segment_data?: {
    start: number;
    end: number;
    effect: string;
    intensity: number;
  };
}

type AppState =
  | "upload"
  | "preview"
  | "editing"
  | "completed"
  | "multi_video_preview";

// Helper function to handle LLM edit with retry logic
const callLLMEditWithRetry = async (
  llmUrl: string,
  requestBody: Record<string, unknown>
) => {
  const response = await fetch(llmUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      Pragma: "no-cache",
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("🔍 [DEBUG] LLM Error response:", errorText);

    // If it's a "Not Found" error, wait a bit more and retry once
    if (
      response.status === 404 &&
      errorText.includes("Analysis results not found")
    ) {
      console.log(
        "🔍 [DEBUG] Analysis results not found, waiting 3 more seconds and retrying..."
      );
      await new Promise((resolve) => setTimeout(resolve, 3000)); // Wait 3 more seconds

      const retryResponse = await fetch(llmUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!retryResponse.ok) {
        const retryErrorText = await retryResponse.text();
        console.error("🔍 [DEBUG] LLM Retry Error response:", retryErrorText);
        throw new Error(
          `LLM analysis failed after retry: ${retryResponse.statusText}`
        );
      }

      console.log("🔍 [DEBUG] LLM retry successful!");
      return retryResponse;
    } else {
      throw new Error(`LLM analysis failed: ${response.statusText}`);
    }
  }

  return response;
};

export default function Home() {
  const [appState, setAppState] = useState<
    | "upload"
    | "uploading"
    | "processing"
    | "completed"
    | "error"
    | "preview"
    | "editing"
    | "multi_video_preview"
  >("upload");
  const [uploadProgress, setUploadProgress] = useState<string>("");
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);
  const [error, setError] = useState<string>("");
  const [projectId, setProjectId] = useState<string>("");
  const [outputVideoUrl, setOutputVideoUrl] = useState<string>("");
  const [projectStatus, setProjectStatus] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [timelineSegments, setTimelineSegments] = useState<
    Array<{
      start: number;
      end: number;
      source_video_id: string;
      effects: string[];
      transition_in?: string;
      transition_out?: string;
      speed: number;
      volume: number;
      effectCustomizations?: { [key: string]: unknown };
      ai_recommendations?: {
        segment_reasoning: string;
        transition_reasoning: string;
        effects_reasoning: string;
        arrangement_reasoning: string;
        confidence_score: number;
        alternative_suggestions: string[];
      };
    }>
  >([]);
  const [llmSuggestions, setLlmSuggestions] = useState<LLMSuggestion[]>([]);
  const [useEnhancedEditor, setUseEnhancedEditor] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [enableCrossAnalysis, setEnableCrossAnalysis] = useState(false);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.8);
  const [chunkingStrategy, setChunkingStrategy] = useState<
    "scene" | "action" | "audio" | "content"
  >("scene");
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<{
    success: boolean;
    message: string;
    timestamp: string;
    video_id: string;
    upload_url: string;
    expires_at: string;
    metadata?: {
      filename: string;
      size: number;
      content_type?: string;
      duration?: number;
    };
  } | null>(null);
  const [editDecision, setEditDecision] = useState<{
    video_id: string;
    style: string;
    segments: Array<{
      start: number;
      end: number;
      tags: string[];
      transition?: string;
      transition_duration?: number;
      speed?: number;
      transition_in?: string;
      transition_out?: string;
    }>;
    notes?: string;
    edit_scale: number;
  } | null>(null);
  const [sourceVideos, setSourceVideos] = useState<
    Array<{
      id: string;
      url: string;
      name: string;
      duration: number;
    }>
  >([]);

  // Authentication state
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false); // Start closed, let useEffect determine
  const [currentUser, setCurrentUser] = useState<OAuthUser | null>(null);

  // Authentication handlers
  const handleAuthSuccess = (user: OAuthUser) => {
    setCurrentUser(user);
    setIsAuthModalOpen(false);
    console.log("User authenticated:", user);
  };

  const handleSignOut = () => {
    setCurrentUser(null);
    localStorage.removeItem("auth_token");
    localStorage.removeItem("user");
    console.log("User signed out");
  };

  // Global error handler for debugging
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      console.error("🔧 [DEBUG] Global error caught:", event.error);
      console.error("🔧 [DEBUG] Error message:", event.message);
      console.error("🔧 [DEBUG] Error filename:", event.filename);
      console.error("🔧 [DEBUG] Error lineno:", event.lineno);
      console.error("🔧 [DEBUG] Error colno:", event.colno);
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error("🔧 [DEBUG] Unhandled promise rejection:", event.reason);
    };

    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      console.log("🔧 [DEBUG] Page is about to unload/navigate");
      console.log("🔧 [DEBUG] Current URL:", window.location.href);
    };

    const handlePopState = (event: PopStateEvent) => {
      console.log("🔧 [DEBUG] Browser back/forward navigation detected");
      console.log("🔧 [DEBUG] New URL:", window.location.href);
    };

    window.addEventListener("error", handleError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);
    window.addEventListener("beforeunload", handleBeforeUnload);
    window.addEventListener("popstate", handlePopState);

    return () => {
      window.removeEventListener("error", handleError);
      window.removeEventListener(
        "unhandledrejection",
        handleUnhandledRejection
      );
      window.removeEventListener("beforeunload", handleBeforeUnload);
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  // Check for existing authentication on page load
  useEffect(() => {
    const token = localStorage.getItem("auth_token");
    const userStr = localStorage.getItem("user");

    console.log("🔍 Checking authentication on page load...");
    console.log("Token exists:", !!token);
    console.log("User data exists:", !!userStr);

    if (token && userStr) {
      try {
        const user = JSON.parse(userStr);
        setCurrentUser(user);
        setIsAuthModalOpen(false);
        console.log("✅ User already authenticated:", user);
      } catch (error) {
        console.error("❌ Error parsing stored user data:", error);
        localStorage.removeItem("auth_token");
        localStorage.removeItem("user");
        setIsAuthModalOpen(true);
      }
    } else {
      console.log("❌ No authentication found, showing login modal");
      setIsAuthModalOpen(true);
    }
  }, []);

  // Debug effect to log authentication state changes
  useEffect(() => {
    console.log("🔧 Authentication state changed:", {
      currentUser: currentUser ? "authenticated" : "not authenticated",
      isAuthModalOpen,
      hasToken: !!localStorage.getItem("auth_token"),
      hasUser: !!localStorage.getItem("user"),
    });
  }, [currentUser, isAuthModalOpen]);

  useEffect(() => {
    // If there is a currentProjectId and timelineSegments are empty, fetch timeline
    if (
      currentProjectId &&
      (!timelineSegments || timelineSegments.length === 0)
    ) {
      (async () => {
        try {
          const response = await fetch(
            `/api/v1/multi-video/projects/${currentProjectId}/timeline`
          );
          if (response.ok) {
            const data = await response.json();
            if (data.segments && Array.isArray(data.segments)) {
              setTimelineSegments(
                data.segments.map((segment: any) => ({
                  start: segment.start_time,
                  end: segment.end_time,
                  source_video_id: segment.source_video_id,
                  video_url: segment.video_url, // Add video URL for original video
                  stream_url: segment.stream_url, // Add stream URL for processed video
                  effects: segment.effects || [],
                  transition_in: segment.transition_in,
                  transition_out: segment.transition_out,
                  speed: segment.speed || 1.0,
                  volume: 1.0,
                  effectCustomizations: segment.effectCustomizations || {},
                }))
              );
            }
          }
        } catch (err) {
          console.error(
            "Failed to fetch timeline for project",
            currentProjectId,
            err
          );
        }
      })();
    }
  }, [currentProjectId]);

  const handleAdvancedUpload = async (params: UploadParams) => {
    console.log("🔧 [DEBUG] ===== HANDLE ADVANCED UPLOAD START =====");
    console.log("🔧 [DEBUG] Current window location:", window.location.href);
    console.log(
      "🔧 [DEBUG] Current window pathname:",
      window.location.pathname
    );
    console.log("🔧 [DEBUG] Upload params received:", params);

    setIsUploading(true);
    setUploadProgress("Starting upload...");
    setAnalysisProgress(0);
    setVideoFile(params.file);

    try {
      console.log("🔧 [DEBUG] Uploading with params:", params);

      // Get the base URL from API client
      console.log("🔧 [DEBUG] Getting base URL from API client...");
      const { apiClient } = await import("@/lib/api");
      const baseURL = await apiClient.getCurrentBackendURL();
      console.log("🔧 [DEBUG] Base URL obtained:", baseURL);

      // Step 1: Upload the video file
      console.log("🔧 [DEBUG] ===== STEP 1: UPLOAD VIDEO FILE =====");
      setUploadProgress("Uploading video file...");
      const formData = new FormData();
      formData.append("file", params.file);
      console.log("🔧 [DEBUG] FormData created with file:", params.file.name);

      const uploadUrl = `${baseURL}/api/v1/videos/upload-direct`;
      console.log("🔧 [DEBUG] Upload URL:", uploadUrl);
      console.log("🔧 [DEBUG] About to make upload request...");

      const uploadResponse = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
      });

      console.log("🔧 [DEBUG] Upload response received:", {
        status: uploadResponse.status,
        statusText: uploadResponse.statusText,
        ok: uploadResponse.ok,
      });

      if (!uploadResponse.ok) {
        console.error(
          "🔧 [DEBUG] Upload failed with status:",
          uploadResponse.status
        );
        console.error(
          "🔧 [DEBUG] Upload failed with statusText:",
          uploadResponse.statusText
        );
        const errorText = await uploadResponse.text();
        console.error("🔧 [DEBUG] Upload error response body:", errorText);
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }

      const uploadResult = await uploadResponse.json();
      console.log("🔧 [DEBUG] Upload result:", uploadResult);

      setUploadedVideo(uploadResult);
      setUploadProgress("Video uploaded successfully!");

      // Step 2: Analyze the video
      setUploadProgress("Analyzing video...");
      setAnalysisProgress(10);

      const analyzeResponse = await fetch(
        `${baseURL}/api/v1/videos/${uploadResult.video_id}/analyze`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            video_url: uploadResult.upload_url,
            template_type: params.templateType,
            analysis_options: {
              edit_style: params.editStyle,
              quality_preset: params.qualityPreset,
            },
          }),
        }
      );

      if (!analyzeResponse.ok) {
        throw new Error(`Analysis failed: ${analyzeResponse.statusText}`);
      }

      const analyzeResult = await analyzeResponse.json();
      console.log("Analysis result:", analyzeResult);
      setAnalysisProgress(50);

      // Step 3: Get LLM suggestions
      setUploadProgress("Generating AI suggestions...");
      setAnalysisProgress(70);

      // Add a small delay to ensure analysis results are fully saved
      await new Promise((resolve) => setTimeout(resolve, 2000)); // Wait 2 seconds

      const llmUrl = `${baseURL}/api/v1/videos/${uploadResult.video_id}/enhanced_llm_edit`;
      console.log("🔍 [DEBUG] Calling LLM endpoint:", llmUrl);
      console.log("🔍 [DEBUG] Video ID:", uploadResult.video_id);
      console.log("🔍 [DEBUG] Base URL:", baseURL);

      const llmResponse = await fetch(llmUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          edit_scale: 0.7,
          style_preferences: {
            style: params.editStyle,
            template_type: params.templateType,
            quality_preset: params.qualityPreset,
          },
          target_duration: null,
          dry_run: false,
        }),
      });

      console.log("🔍 [DEBUG] LLM Response status:", llmResponse.status);
      console.log(
        "🔍 [DEBUG] LLM Response statusText:",
        llmResponse.statusText
      );

      if (!llmResponse.ok) {
        const errorText = await llmResponse.text();
        console.error("🔍 [DEBUG] LLM Error response:", errorText);
        throw new Error(`LLM analysis failed: ${llmResponse.statusText}`);
      }

      const llmResult = await llmResponse.json();
      console.log("LLM result:", llmResult);
      setAnalysisProgress(90);

      // Step 4: Wait for the enhanced LLM edit job to complete
      setUploadProgress("Processing enhanced edit...");

      // Poll for job completion
      let jobCompleted = false;
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds timeout

      while (!jobCompleted && attempts < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, 1000)); // Wait 1 second

        const statusResponse = await fetch(
          `${baseURL}/api/v1/videos/${uploadResult.video_id}/status`
        );

        if (statusResponse.ok) {
          const statusResult = await statusResponse.json();
          const jobStatus = statusResult.job?.status;
          const jobProgress = statusResult.job?.progress || 0;

          console.log("Job status:", jobStatus, "Progress:", jobProgress);
          setAnalysisProgress(90 + jobProgress * 0.1); // Scale progress from 90-100

          if (jobStatus === "completed") {
            jobCompleted = true;
            console.log("Enhanced LLM edit job completed!");
          } else if (jobStatus === "failed") {
            throw new Error("Enhanced LLM edit job failed");
          }
        }

        attempts++;
      }

      if (!jobCompleted) {
        throw new Error("Enhanced LLM edit job timed out");
      }

      // Step 5: Get the edit decision
      setUploadProgress("Finalizing edit decision...");
      const editDecisionResponse = await fetch(
        `${baseURL}/api/v1/videos/${uploadResult.video_id}/timeline`
      );

      if (!editDecisionResponse.ok) {
        throw new Error(
          `Timeline retrieval failed: ${editDecisionResponse.statusText}`
        );
      }

      const editDecisionResult = await editDecisionResponse.json();
      console.log("Edit decision result:", editDecisionResult);

      setEditDecision(editDecisionResult);
      setAnalysisProgress(100);
      setUploadProgress("Analysis completed!");

      // Step 6: Move to preview state
      setTimeout(() => {
        setAppState("preview");
        setIsUploading(false);
      }, 1000);
    } catch (error) {
      console.error("Upload process failed:", error);
      setUploadProgress(
        `Error: ${error instanceof Error ? error.message : "Unknown error"}`
      );
      setIsUploading(false);
    }
  };

  const handleMultiVideoUpload = async () => {
    console.log("🔧 [DEBUG] ===== HANDLE MULTI VIDEO UPLOAD START =====");
    console.log("🔧 [DEBUG] Current window location:", window.location.href);
    console.log(
      "🔧 [DEBUG] Current window pathname:",
      window.location.pathname
    );
    console.log("🔧 [DEBUG] Selected files:", selectedFiles?.length || 0);

    if (!selectedFiles || selectedFiles.length === 0) {
      console.log("🔧 [DEBUG] No files selected for multi-video upload");
      setError("Please select at least one video file");
      return;
    }

    setIsUploading(true);
    setUploadProgress("Starting multi-video upload...");
    setError("");

    try {
      console.log("🔧 [DEBUG] ===== MULTI-VIDEO UPLOAD PROCESS =====");
      console.log(
        "🚀 Starting multi-video upload with",
        selectedFiles.length,
        "files"
      );

      // Create multi-video project
      const projectResponse = await multiVideoApi.createProject(
        selectedFiles,
        `Multi-Video Project ${new Date().toISOString()}`,
        {
          enableCrossAnalysis: enableCrossAnalysis,
          similarityThreshold: similarityThreshold,
          chunkingStrategy: chunkingStrategy,
        }
      );

      console.log("✅ Multi-video project created:", projectResponse);
      setCurrentProjectId(projectResponse.project_id);

      console.log(
        "🔄 Starting polling for project:",
        projectResponse.project_id
      );

      // Don't start cross-video analysis immediately - wait for individual analysis jobs to complete
      // The backend will handle cross-video analysis automatically when all individual jobs are done
      console.log(
        "⏳ Waiting for individual video analysis to complete before cross-video analysis..."
      );
      setAnalysisProgress(10);

      // Poll for completion with detailed progress
      let attempts = 0;
      const maxAttempts = 600; // 300 seconds (5 minutes) with 500ms intervals (increased for longer processing)

      const pollStatus = async () => {
        try {
          console.log(
            `🔄 Polling attempt ${attempts + 1} for project:`,
            projectResponse.project_id
          );
          console.log("🔄 Polling function started");
          const status = await multiVideoApi.getProjectStatus(
            projectResponse.project_id
          );
          console.log("📊 Project status:", status);
          console.log("🔍 Current app state:", appState);
          console.log("🔍 Editing completed check:", status.editing_completed);
          console.log(
            "🔍 Status.editing_completed type:",
            typeof status.editing_completed
          );
          console.log(
            "🔍 Status.editing_completed value:",
            status.editing_completed
          );

          // Use backend progress directly
          if (status.progress !== undefined) {
            setAnalysisProgress(status.progress);
            console.log(`📊 Progress updated to: ${status.progress}%`);
          } else {
            // No fallback - require backend to provide progress
            console.error("❌ Backend did not provide progress");
            throw new Error("Backend progress not available");
          }

          // Show appropriate status messages
          const totalVideos = status.video_ids?.length || 0;

          if (status.status === "analyzing") {
            setUploadProgress(
              `Analyzing videos: ${status.analysis_completed}/${totalVideos} completed`
            );
            console.log(
              `⏳ Analysis in progress: ${status.analysis_completed}/${totalVideos} videos analyzed`
            );

            // Check if all individual analysis jobs are completed and trigger cross-analysis
            console.log(
              `🔍 Checking cross-analysis trigger: analysis_completed=${status.analysis_completed}, totalVideos=${totalVideos}, cross_analysis_completed=${status.cross_analysis_completed}`
            );

            // Show progress updates based on current status
            if (status.status === "analyzing") {
              if (status.analysis_completed < totalVideos) {
                setUploadProgress(
                  `Analyzing videos: ${status.analysis_completed}/${totalVideos} completed`
                );
                console.log(
                  `⏳ Analysis in progress: ${status.analysis_completed}/${totalVideos} videos analyzed`
                );
              } else if (!status.cross_analysis_completed) {
                setUploadProgress("Cross-video analysis in progress...");
                console.log("🔄 Cross-analysis in progress...");
              } else if (
                status.cross_analysis_completed &&
                !status.editing_completed
              ) {
                setUploadProgress(
                  "Cross-analysis completed, processing edit..."
                );
                console.log(
                  "✅ Cross-analysis completed, editing in progress..."
                );
              }
            } else if (status.status === "editing") {
              setUploadProgress("Processing multi-video edit...");
              console.log(
                "🎬 Project is in editing state, waiting for completion..."
              );
            }
          } else if (status.status === "editing") {
            setUploadProgress("Processing multi-video edit...");
            console.log(
              "🎬 Project is in editing state, waiting for completion..."
            );
          }
          // Check if cross-analysis is completed and editing is in progress
          if (status.cross_analysis_completed && !status.editing_completed) {
            setUploadProgress("Cross-analysis completed, processing edit...");
            console.log("🔄 Cross-analysis completed, editing in progress...");
          }

          // Check if everything is completed and transition to preview page
          if (
            status.status === "completed" &&
            status.output_video_url &&
            status.editing_completed === true
          ) {
            setAnalysisProgress(100);
            // Don't set outputVideoUrl - use proxy URLs from timeline segments instead
            console.log("🎉 Multi-video editing completed!");
            console.log("📊 Status details:", {
              editing_completed: status.editing_completed,
              progress: status.progress,
              has_metadata: !!status.metadata,
              has_llm_plan: !!status.metadata?.enhanced_llm_plan_json,
              app_state: "will be set to preview",
            });

            // Get the timeline data from the backend (same as single video workflow)
            try {
              console.log(
                "📋 Getting timeline data for project:",
                projectResponse.project_id
              );
              console.log(
                "📋 About to call getProjectTimeline for project:",
                projectResponse.project_id
              );
              const timelineResponse = await multiVideoApi.getProjectTimeline(
                projectResponse.project_id
              );
              console.log("📋 Timeline response:", timelineResponse);
              console.log(
                "📋 Timeline response type:",
                typeof timelineResponse
              );
              console.log(
                "📋 Timeline response keys:",
                Object.keys(timelineResponse || {})
              );

              if (
                timelineResponse.segments &&
                timelineResponse.segments.length > 0
              ) {
                // Convert multi-video segments to single video edit decision format
                const segments = timelineResponse.segments.map(
                  (segment: any) => ({
                    start: segment.start_time,
                    end: segment.end_time,
                    tags: segment.effects || [], // Convert effects to tags
                    transition: segment.transition_in,
                    transition_duration: 0.5,
                    speed: segment.speed || 1.0,
                    transition_in: segment.transition_in,
                    transition_out: segment.transition_out,
                  })
                );

                // Create edit decision in single video format
                const editDecision = {
                  video_id: "multi_video_combined",
                  style: timelineResponse.style || "cinematic",
                  segments: segments,
                  notes: "Multi-video combined edit",
                  edit_scale: 1.0,
                };

                setEditDecision(editDecision);
                console.log(
                  "✅ Set edit decision for multi-video:",
                  editDecision
                );

                // Also set timeline segments for multi-video preview workflow
                const multiVideoSegments = timelineResponse.segments.map(
                  (segment: any) => ({
                    start: segment.start_time,
                    end: segment.end_time,
                    source_video_id: segment.source_video_id,
                    video_url: segment.video_url, // Add video URL for original video
                    stream_url: segment.stream_url, // Add stream URL for processed video
                    effects: segment.effects || [],
                    transition_in: segment.transition_in,
                    transition_out: segment.transition_out,
                    speed: segment.speed || 1.0,
                    volume: 1.0,
                    effectCustomizations: segment.effectCustomizations || {},
                  })
                );

                // Set timeline segments and LLM suggestions separately
                setTimelineSegments(multiVideoSegments);

                // Set LLM suggestions for the multi-video preview
                if (timelineResponse.llm_suggestions) {
                  setLlmSuggestions(timelineResponse.llm_suggestions);
                  console.log(
                    "✅ Set LLM suggestions:",
                    timelineResponse.llm_suggestions
                  );
                }
                console.log(
                  "✅ Set timeline segments for multi-video preview:",
                  multiVideoSegments
                );

                // Set source videos for multi-video preview workflow
                const sourceVideosData =
                  status.video_ids?.map((videoId: string, index: number) => ({
                    id: videoId,
                    url: `/api/v1/videos/${videoId}/download?video_type=original`, // Use relative backend download endpoint
                    name: `Video ${index + 1}`,
                    duration: 10.0, // Placeholder duration
                  })) || [];

                setSourceVideos(sourceVideosData);
                console.log(
                  "✅ Set source videos for multi-video preview:",
                  sourceVideosData
                );

                // Set video file and uploaded video for preview (use proxy streaming endpoint)
                if (status.output_video_url) {
                  console.log(
                    "🎬 Using processed video from backend:",
                    status.output_video_url
                  );

                  // Use the first timeline segment's stream URL instead of direct S3 URL
                  const firstSegment = timelineResponse.segments?.[0];
                  if (firstSegment?.stream_url) {
                    const proxyVideoUrl = `http://localhost:8000${firstSegment.stream_url}`;
                    console.log("🎬 Using proxy video URL:", proxyVideoUrl);

                    // Fetch the processed video from the proxy endpoint
                    try {
                      const videoResponse = await fetch(proxyVideoUrl);
                      const videoBlob = await videoResponse.blob();
                      const videoFile = new File(
                        [videoBlob],
                        "processed_multi_video.mp4",
                        { type: "video/mp4" }
                      );

                      setVideoFile(videoFile);
                      setUploadedVideo({
                        success: true,
                        message: "Video uploaded successfully",
                        timestamp: new Date().toISOString(),
                        video_id: "multi_video_combined",
                        upload_url: URL.createObjectURL(videoFile),
                        expires_at: new Date(
                          Date.now() + 3600000
                        ).toISOString(), // 1 hour expiration
                        metadata: {
                          filename: "processed_multi_video.mp4",
                          size: videoBlob.size,
                          content_type: "video/mp4",
                          duration: 10.0, // Placeholder duration
                        },
                      });

                      console.log(
                        "✅ Set processed video file for preview using proxy URL"
                      );
                    } catch (error) {
                      console.error(
                        "❌ Failed to fetch processed video from proxy:",
                        error
                      );
                      throw new Error(
                        "Failed to fetch processed video from backend proxy"
                      );
                    }
                  } else {
                    console.error(
                      "❌ No stream URL found in timeline segments"
                    );
                    throw new Error("No stream URL provided by backend");
                  }
                } else {
                  console.error("❌ No output video URL found");
                  throw new Error("No output video URL provided by backend");
                }
              } else {
                console.warn("⚠️ No segments found in timeline response");
              }
            } catch (error) {
              console.error("❌ Failed to get timeline data:", error);
              throw new Error("Failed to get timeline data from backend");
            }

            // Transition to multi-video preview state to show the timeline editor
            console.log("🔄 Transitioning to multi-video preview state...");
            console.log("🔄 Current app state before transition:", appState);
            setIsUploading(false); // Set uploading to false when process is complete
            setAppState("multi_video_preview");
            console.log("🔄 setAppState('multi_video_preview') called");
            return;
          } else if (status.status === "failed") {
            setError("Multi-video editing failed");
            setIsUploading(false); // Set uploading to false on failure
            return;
          }

          attempts++;
          console.log(
            `🔄 Polling attempt ${attempts}/${maxAttempts} (${Math.round(
              attempts * 0.5
            )}s elapsed)`
          );

          // Log detailed status every 10 attempts
          if (attempts % 10 === 0) {
            console.log(`📊 Status check #${attempts}:`, {
              status: status.status,
              analysis_completed: status.analysis_completed,
              cross_analysis_completed: status.cross_analysis_completed,
              editing_completed: status.editing_completed,
              output_video_url: status.output_video_url
                ? "Available"
                : "Not available",
            });

            // Update progress based on backend's progress value
            if (status.progress !== undefined) {
              setAnalysisProgress(status.progress);
              console.log(`📊 Progress updated to: ${status.progress}%`);
            } else {
              // No fallback - require backend to provide progress
              console.error("❌ Backend did not provide progress");
              throw new Error("Backend progress not available");
            }
          }

          if (attempts < maxAttempts) {
            setTimeout(pollStatus, 500);
          } else {
            console.error(
              "⏰ Multi-video editing timed out after",
              maxAttempts,
              "attempts (5 minutes)"
            );
            console.error("📊 Final status:", status);
            setError("Multi-video editing timed out after 5 minutes");
            setIsUploading(false); // Set uploading to false on timeout
          }
        } catch (error) {
          console.error("❌ Error polling project status:", error);
          setError("Failed to check project status");
          setIsUploading(false); // Set uploading to false on error
        }
      };

      pollStatus();
    } catch (error) {
      console.error("❌ Multi-video upload failed:", error);
      setError(
        `Multi-video upload failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
      setIsUploading(false);
    }
    // Don't set isUploading to false here - let it stay true during polling
  };

  const handlePreviewProceed = () => {
    setAppState("editing");
  };

  const handlePreviewBack = () => {
    setAppState("upload");
  };

  const handleAdjustSegment = (
    segmentIndex: number,
    adjustments: Partial<EditDecisionSegment> | EditDecisionSegment
  ) => {
    console.log("[DEBUG] handleAdjustSegment called", {
      segmentIndex,
      adjustments,
    });
    if (editDecision) {
      const updatedSegments = [...editDecision.segments];
      // If segmentIndex is at the end of the array, add a new segment
      if (segmentIndex === updatedSegments.length) {
        updatedSegments.push(adjustments as EditDecisionSegment);
        console.log("[DEBUG] Added new segment", updatedSegments);
      } else {
        // Update existing segment
        updatedSegments[segmentIndex] = {
          ...updatedSegments[segmentIndex],
          ...adjustments,
        };
        console.log("[DEBUG] Updated segment", updatedSegments);
      }
      setEditDecision({
        ...editDecision,
        segments: updatedSegments,
      });
      console.log("[DEBUG] setEditDecision called", { updatedSegments });
    }
  };

  const handleReset = () => {
    setAppState("upload");
    setUploadedVideo(null);
    setVideoFile(null);
    setEditDecision(null);
    setLlmSuggestions([]);
  };

  const isMultiVideo =
    Array.isArray(timelineSegments) &&
    timelineSegments.length > 0 &&
    new Set(timelineSegments.map((s: any) => s.source_video_id)).size > 1;

  // Debug logging for segment count
  console.log("🔧 [DEBUG] isMultiVideo:", isMultiVideo);
  console.log("🔧 [DEBUG] timelineSegments length:", timelineSegments.length);
  console.log("🔧 [DEBUG] timelineSegments:", timelineSegments);
  if (timelineSegments.length > 0) {
    console.log(
      "🔧 [DEBUG] Unique source_video_ids:",
      new Set(timelineSegments.map((s: any) => s.source_video_id))
    );
  }

  return (
    <main>
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 relative overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%239C92AC%22%20fill-opacity%3D%220.1%22%3E%3Ccircle%20cx%3D%227%22%20cy%3D%227%22%20r%3D%227%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-20"></div>

        {/* Floating Orbs */}
        <motion.div
          className="absolute top-20 left-20 w-32 h-32 bg-purple-500/20 rounded-full blur-xl"
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear",
          }}
        />
        <motion.div
          className="absolute bottom-20 right-20 w-40 h-40 bg-blue-500/20 rounded-full blur-xl"
          animate={{
            x: [0, -80, 0],
            y: [0, 60, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "linear",
          }}
        />

        {/* Main Content */}
        <div className="relative z-10 w-full h-full">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="w-full bg-white/10 backdrop-blur-md border-b border-white/20 px-12 py-6 z-10"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-8">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                  className="text-purple-400 w-8 h-8 flex-shrink-0"
                >
                  <Sparkles />
                </motion.div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent whitespace-nowrap">
                  AI Video Editor
                </h1>
                <motion.div
                  animate={{ rotate: -360 }}
                  transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                  className="text-blue-400 w-8 h-8 flex-shrink-0"
                >
                  <Video />
                </motion.div>
                <div className="flex items-center text-base text-gray-300 ml-8">
                  <span>Transform your videos with AI-powered editing</span>
                </div>
              </div>

              {/* Authentication UI */}
              <div className="flex items-center gap-4">
                {currentUser ? (
                  <div className="flex items-center gap-3">
                    <span className="text-white text-sm">
                      Welcome, {currentUser.name}
                    </span>
                    <button
                      onClick={handleSignOut}
                      className="px-4 py-2 text-sm bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                    >
                      Sign Out
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setIsAuthModalOpen(true)}
                    className="px-6 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg font-medium transition-all duration-200 hover:scale-105"
                  >
                    Sign In
                  </button>
                )}
              </div>
            </div>
          </motion.div>

          {/* Feature Cards - Only show on upload state */}
          {appState === "upload" && (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="grid md:grid-cols-3 gap-6 mb-12"
            >
              {[
                {
                  icon: Zap,
                  title: "Lightning Fast",
                  desc: "AI processing in seconds",
                },
                {
                  icon: Star,
                  title: "Professional Quality",
                  desc: "Hollywood-grade editing",
                },
                {
                  icon: Sparkles,
                  title: "Smart Automation",
                  desc: "Intelligent scene detection",
                },
              ].map((feature, index) => (
                <motion.div
                  key={index}
                  whileHover={{ scale: 1.05, y: -5 }}
                  className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 hover:border-purple-400/50 transition-all duration-300"
                >
                  <feature.icon className="text-purple-400 w-8 h-8 mb-4" />
                  <h3 className="text-white font-semibold text-lg mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-300 text-sm">{feature.desc}</p>
                </motion.div>
              ))}
            </motion.div>
          )}

          {/* Main Interface */}
          <div className="w-full h-full">
            {appState === "upload" && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                <div className="rounded-lg border bg-card text-card-foreground shadow-sm w-full max-w-2xl mx-auto">
                  <AdvancedUploadPanel
                    onUpload={handleAdvancedUpload}
                    onMultiUpload={(files) => {
                      setSelectedFiles(files);
                      handleMultiVideoUpload();
                    }}
                    isUploading={isUploading}
                    uploadProgress={uploadProgress}
                    analysisProgress={analysisProgress}
                  />
                </div>

                {/* Enhanced Editor Toggle */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-semibold text-white">
                          Editor Mode
                        </h3>
                        <p className="text-sm text-gray-300">
                          Choose between standard and enhanced editing modes
                        </p>
                      </div>
                      <div className="flex items-center gap-4">
                        <button
                          onClick={() => setUseEnhancedEditor(false)}
                          className={`px-4 py-2 rounded-lg transition-all duration-300 flex items-center gap-2 ${
                            !useEnhancedEditor
                              ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg"
                              : "bg-white/20 text-gray-300 hover:bg-white/30"
                          }`}
                        >
                          <Video className="w-4 h-4" />
                          Standard
                        </button>
                        <button
                          onClick={() => setUseEnhancedEditor(true)}
                          className={`px-4 py-2 rounded-lg transition-all duration-300 flex items-center gap-2 ${
                            useEnhancedEditor
                              ? "bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg"
                              : "bg-white/20 text-gray-300 hover:bg-white/30"
                          }`}
                        >
                          <Sparkles className="w-4 h-4" />
                          Enhanced
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              </motion.div>
            )}

            {isMultiVideo ? (
              <div className="min-h-screen">
                <MultiVideoPreviewPage
                  sourceVideos={sourceVideos}
                  timelineSegments={timelineSegments}
                  outputVideoUrl={outputVideoUrl}
                  llmSuggestions={llmSuggestions}
                  onUpdateSegment={(segmentIndex: number, updates: any) => {
                    // Update the timeline segments
                    const updatedSegments = [...timelineSegments];
                    updatedSegments[segmentIndex] = {
                      ...updatedSegments[segmentIndex],
                      ...updates,
                    };
                    setTimelineSegments(updatedSegments);
                  }}
                  onBack={() => {
                    setAppState("upload");
                    setCurrentProjectId(null);
                    setTimelineSegments([]);
                    setSourceVideos([]);
                  }}
                  onProceed={() => {
                    // Handle final proceed action - download the video
                    if (outputVideoUrl) {
                      // Create a download link for the final video
                      const a = document.createElement("a");
                      a.href = outputVideoUrl;
                      a.download = "multi-video-edited.mp4";
                      a.target = "_blank";
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                    } else {
                      console.log("No output video URL available for download");
                    }
                  }}
                />
              </div>
            ) : appState === "preview" &&
              editDecision &&
              uploadedVideo &&
              videoFile &&
              !currentProjectId ? (
              <>
                {console.log(
                  "🔧 PreviewPage - convertedVideoUrl:",
                  uploadedVideo.upload_url
                )}
                <PreviewPage
                  videoFile={videoFile}
                  convertedVideoUrl={uploadedVideo.upload_url}
                  editDecision={editDecision}
                  onProceed={handlePreviewProceed}
                  onAdjust={handleAdjustSegment}
                  onBack={handlePreviewBack}
                  llmSuggestions={llmSuggestions}
                />
              </>
            ) : appState === "preview" ? (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400 mx-auto mb-4"></div>
                <p className="text-gray-300">Loading video preview...</p>
                <div className="text-xs text-gray-500 mt-2">
                  Debug: videoFile={!!videoFile}, uploadedVideo=
                  {!!uploadedVideo}, editDecision={!!editDecision}
                </div>
              </div>
            ) : null}

            {appState === "editing" &&
              uploadedVideo &&
              (useEnhancedEditor ? (
                <EnhancedVideoEditor
                  uploadedVideo={uploadedVideo}
                  onReset={handleReset}
                />
              ) : (
                <VideoEditor
                  uploadedVideo={uploadedVideo}
                  onReset={handleReset}
                />
              ))}

            {appState === "completed" && !isMultiVideo && (
              <div className="min-h-screen">
                <MultiVideoPreviewPage
                  sourceVideos={sourceVideos}
                  timelineSegments={timelineSegments}
                  outputVideoUrl={outputVideoUrl}
                  onUpdateSegment={(segmentIndex: number, updates: any) => {
                    // Update the timeline segments
                    const updatedSegments = [...timelineSegments];
                    updatedSegments[segmentIndex] = {
                      ...updatedSegments[segmentIndex],
                      ...updates,
                    };
                    setTimelineSegments(updatedSegments);
                  }}
                  onBack={() => {
                    setAppState("upload");
                    setCurrentProjectId(null);
                    setTimelineSegments([]);
                    setSourceVideos([]);
                  }}
                  onProceed={() => {
                    // Handle final proceed action
                    console.log("Final proceed action");
                  }}
                />
              </div>
            )}
          </div>

          {/* Footer */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-center mt-16 text-gray-400"
          >
            <div className="flex items-center justify-center gap-4 mb-4">
              <a
                href="https://organic-swim-production.up.railway.app/docs"
                className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-6 py-3 rounded-full font-medium transition-all duration-300 hover:scale-105 hover:shadow-lg"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Github className="w-4 h-4" />
                API Documentation
              </a>
            </div>

            {/* Backend Status Component */}
            <div className="mt-4">
              <BackendStatus className="text-white" />
            </div>

            <p className="text-sm mt-4">
              Powered by FastAPI • Built with Next.js 15
            </p>
          </motion.div>
        </div>
      </div>

      {/* Environment Switcher */}
      <EnvironmentSwitcher />

      {/* Auth Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onAuthSuccess={handleAuthSuccess}
      />
    </main>
  );
}
