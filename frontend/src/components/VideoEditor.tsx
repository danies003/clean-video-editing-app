"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  Download,
  Settings,
  Zap,
  Wand2,
  CheckCircle,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  apiClient,
  VideoUploadResponse,
  ProcessingJob,
  AdvancedEditRequest,
  CompletedVideoResponse,
} from "@/lib/api";

interface VideoEditorProps {
  uploadedVideo: VideoUploadResponse;
  onReset: () => void;
}

type EditingStep = "analysis" | "editing" | "completed" | "error";

interface EditingState {
  step: EditingStep;
  analysisJob?: ProcessingJob;
  editingJob?: ProcessingJob;
  error?: string;
  resultUrl?: string;
}

export function VideoEditor({ uploadedVideo, onReset }: VideoEditorProps) {
  const [editingState, setEditingState] = useState<EditingState>({
    step: "analysis",
  });
  const [editSettings, setEditSettings] = useState<AdvancedEditRequest>({
    video_id: uploadedVideo.video_id,
    edit_scale: 1.0,
    style_preferences: {
      style: "tiktok", // Add the missing style field
      energy_level: "high",
      transition_style: "dynamic",
      pacing: "fast",
    },
    target_duration: undefined, // Let LLM determine optimal duration based on content
  });

  // Poll job status
  const pollJobStatus = async (
    videoId: string,
    jobType: "analysis" | "editing"
  ) => {
    try {
      const response = await apiClient.getJobStatus(videoId);

      // Handle both job response format and direct video response format
      let job: ProcessingJob;
      let outputUrl: string | undefined = undefined;

      if ("job" in response && response.job) {
        // Standard job response format
        job = response.job as ProcessingJob;
      } else if ("status" in response && response.status) {
        // Direct video response format (when job is complete and contains full video data)
        const videoResponse = response as CompletedVideoResponse;
        outputUrl = videoResponse.output_url;
        job = {
          job_id: videoResponse.metadata?.rq_job_id || "unknown",
          status: videoResponse.status,
          progress: videoResponse.status === "completed" ? 100 : 50,
          result: {
            output_url: videoResponse.output_url,
            analysis_result: videoResponse.analysis,
            timeline: videoResponse.timeline,
          },
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
      } else {
        // No fallback - require proper job response
        throw new Error("Invalid job response format from backend");
      }

      if (jobType === "analysis") {
        setEditingState((prev) => ({ ...prev, analysisJob: job }));

        if (job.status === "completed") {
          // Start editing automatically after analysis
          startEditing();
        } else if (job.status === "failed") {
          setEditingState((prev) => ({
            ...prev,
            step: "error",
            error: job.error || "Analysis failed",
          }));
        }
      } else {
        setEditingState((prev) => ({ ...prev, editingJob: job }));

        if (job.status === "completed") {
          // Use the backend's download endpoint instead of direct S3 URLs
          const downloadUrl = await apiClient.getVideoDownloadUrl(
            uploadedVideo.video_id
          );
          const finalUrl = job.output_url
            ? `${downloadUrl}?video_type=processed`
            : `${downloadUrl}?video_type=original`;

          setEditingState((prev) => ({
            ...prev,
            step: "completed",
            resultUrl: finalUrl,
          }));
        } else if (job.status === "failed") {
          setEditingState((prev) => ({
            ...prev,
            step: "error",
            error: job.error || "Editing failed",
          }));
        }
      }
    } catch (error) {
      console.error("Error polling job status:", error);
    }
  };

  // Start analysis
  const startAnalysis = async () => {
    try {
      console.log("🚀 Starting analysis for video:", uploadedVideo.video_id);
      console.log("📤 Upload URL:", uploadedVideo.upload_url);

      const response = await apiClient.analyzeVideo(uploadedVideo.video_id, {
        template_type: "beat_match",
        analysis_options: {
          edit_style: "tiktok",
          quality_preset: "high",
        },
        video_url: uploadedVideo.upload_url,
      });

      console.log(
        "✅ Analysis started successfully, job_id:",
        response.job?.job_id
      );

      setEditingState((prev) => ({
        ...prev,
        analysisJob: {
          job_id: response.job?.job_id || "unknown",
          status: "pending",
          progress: 0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      }));
    } catch (error) {
      console.error("❌ Analysis start failed:", error);
      console.error("Error details:", {
        message: error instanceof Error ? error.message : "Unknown error",
        stack: error instanceof Error ? error.stack : undefined,
        videoId: uploadedVideo.video_id,
        uploadUrl: uploadedVideo.upload_url,
      });

      setEditingState((prev) => ({
        ...prev,
        step: "error",
        error:
          error instanceof Error ? error.message : "Failed to start analysis",
      }));
    }
  };

  // Start editing
  const startEditing = async () => {
    try {
      setEditingState((prev) => ({ ...prev, step: "editing" }));
      const response = await apiClient.llmEdit(editSettings);
      setEditingState((prev) => ({
        ...prev,
        editingJob: {
          job_id: response.job?.job_id || "unknown",
          status: "pending",
          progress: 0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      }));
    } catch (error) {
      setEditingState((prev) => ({
        ...prev,
        step: "error",
        error:
          error instanceof Error ? error.message : "Failed to start editing",
      }));
    }
  };

  // Poll for job updates
  useEffect(() => {
    let interval: NodeJS.Timeout;

    // Only poll if we have active jobs that are not completed or failed
    const shouldPollAnalysis =
      editingState.analysisJob &&
      ["pending", "processing", "analyzing"].includes(
        editingState.analysisJob.status
      ) &&
      editingState.step === "analysis";

    const shouldPollEditing =
      editingState.editingJob &&
      ["pending", "processing", "editing", "rendering"].includes(
        editingState.editingJob.status
      ) &&
      editingState.step === "editing";

    if (shouldPollAnalysis) {
      console.log(
        "Starting analysis polling for video:",
        uploadedVideo.video_id
      );
      interval = setInterval(() => {
        pollJobStatus(uploadedVideo.video_id, "analysis");
      }, 10000); // Poll every 10 seconds instead of 2 seconds
    } else if (shouldPollEditing) {
      console.log(
        "Starting editing polling for video:",
        uploadedVideo.video_id
      );
      interval = setInterval(() => {
        pollJobStatus(uploadedVideo.video_id, "editing");
      }, 10000); // Poll every 10 seconds instead of 2 seconds
    } else {
      console.log("No active jobs to poll. Current state:", {
        step: editingState.step,
        analysisStatus: editingState.analysisJob?.status,
        editingStatus: editingState.editingJob?.status,
      });
    }

    return () => {
      if (interval) {
        console.log("Clearing polling interval");
        clearInterval(interval);
      }
    };
  }, [
    editingState.analysisJob?.status,
    editingState.editingJob?.status,
    editingState.step,
    uploadedVideo.video_id,
  ]);

  // Start analysis on mount
  useEffect(() => {
    startAnalysis();
  }, []);

  const renderProgressCard = (
    job: ProcessingJob,
    title: string,
    description: string
  ) => {
    // Extract analysis mode from job metadata
    const analysisMode = job.metadata?.analysis_mode;
    const analysisError = job.metadata?.analysis_error;

    const getAnalysisModeBadge = () => {
      if (!analysisMode) return null;

      const modeConfig = {
        real_engine: {
          label: "Real Analysis",
          color: "bg-green-100 text-green-800 border-green-200",
        },
        simple_engine: {
          label: "Simple Analysis",
          color: "bg-yellow-100 text-yellow-800 border-yellow-200",
        },
        mock_analysis: {
          label: "Mock Analysis",
          color: "bg-orange-100 text-orange-800 border-orange-200",
        },
      };

      const config = modeConfig[analysisMode as keyof typeof modeConfig];
      if (!config) return null;

      return (
        <div
          className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${config.color}`}
        >
          {config.label}
        </div>
      );
    };

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-2xl p-6 shadow-lg border"
      >
        <div className="flex items-center space-x-4">
          <div className="relative">
            {job.status === "completed" ? (
              <CheckCircle className="w-8 h-8 text-green-500" />
            ) : job.status === "failed" ? (
              <AlertCircle className="w-8 h-8 text-red-500" />
            ) : (
              <div className="relative w-8 h-8">
                <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
              </div>
            )}
          </div>

          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold text-gray-900">{title}</h3>
              {getAnalysisModeBadge()}
            </div>
            <p className="text-sm text-gray-600">{description}</p>

            {job.status === "processing" && (
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <motion.div
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${job.progress}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {job.progress}% complete
                </p>
              </div>
            )}

            {job.message && (
              <p className="text-xs text-gray-500 mt-1">{job.message}</p>
            )}

            {/* Show analysis mode details when completed */}
            {job.status === "completed" && analysisMode && (
              <div className="mt-2 p-2 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-600">
                  <strong>Analysis Method:</strong>{" "}
                  {analysisMode.replace("_", " ").toUpperCase()}
                </p>
                {analysisError && (
                  <p className="text-xs text-orange-600 mt-1">
                    <strong>Note:</strong> {analysisError}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Video Preview - Only show after editing is completed */}
      {editingState.step === "completed" && editingState.resultUrl && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-6 text-white"
        >
          <div className="aspect-video bg-black rounded-xl mb-4 flex items-center justify-center">
            <div className="text-center space-y-2">
              <CheckCircle className="w-16 h-16 mx-auto text-green-400" />
              <p className="text-gray-400">Edited Video Ready</p>
              <p className="text-sm text-gray-500">
                {uploadedVideo.metadata?.filename || "Unknown file"}
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h2 className="text-xl font-semibold">
                {uploadedVideo.metadata?.filename || "Unknown file"} (Edited)
              </h2>
              <p className="text-gray-400 text-sm">
                Size:{" "}
                {uploadedVideo.metadata?.size
                  ? (uploadedVideo.metadata.size / (1024 * 1024)).toFixed(1)
                  : "Unknown"}{" "}
                MB
                {uploadedVideo.metadata?.duration &&
                  ` • Duration: ${uploadedVideo.metadata.duration}s`}
              </p>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={async () => {
                  try {
                    const response = await fetch(editingState.resultUrl!);
                    const data = await response.json();
                    if (data.download_url) {
                      window.open(data.download_url, "_blank");
                    }
                  } catch (error) {
                    console.error("Download failed:", error);
                  }
                }}
                className="bg-green-600 hover:bg-green-700"
              >
                <Download className="w-4 h-4 mr-2" />
                Download
              </Button>
              <Button variant="outline" onClick={onReset}>
                New Video
              </Button>
            </div>
          </div>
        </motion.div>
      )}

      {/* AI Editing Status & Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-2xl p-6 shadow-lg border"
      >
        <div className="flex items-center space-x-3 mb-6">
          <Wand2 className="w-6 h-6 text-purple-500" />
          <h3 className="text-lg font-semibold">
            {editingState.step === "analysis" && "Analyzing Video"}
            {editingState.step === "editing" && "AI Editing in Progress"}
            {editingState.step === "completed" && "Editing Complete"}
            {editingState.step === "error" && "Error Occurred"}
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">
              Energy Level
            </label>
            <div className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <span className="text-gray-900 font-medium">
                {editSettings.style_preferences?.energy_level === "low" &&
                  "Calm & Relaxed"}
                {editSettings.style_preferences?.energy_level === "medium" &&
                  "Balanced"}
                {editSettings.style_preferences?.energy_level === "high" &&
                  "High Energy"}
              </span>
              <p className="text-xs text-gray-500 mt-1">
                AI will match this energy
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">
              Transition Style
            </label>
            <div className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <span className="text-gray-900 font-medium">
                {editSettings.style_preferences?.transition_style ===
                  "smooth" && "Smooth Crossfades"}
                {editSettings.style_preferences?.transition_style ===
                  "dynamic" && "Dynamic Cuts"}
                {editSettings.style_preferences?.transition_style ===
                  "aggressive" && "Aggressive Edits"}
              </span>
              <p className="text-xs text-gray-500 mt-1">AI transition style</p>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">
              Edit Intensity
            </label>
            <div className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <span className="text-gray-900 font-medium">
                {editSettings.edit_scale === 0.5 && "Subtle"}
                {editSettings.edit_scale === 1.0 && "Moderate"}
                {editSettings.edit_scale === 1.5 && "Dramatic"}
              </span>
              <p className="text-xs text-gray-500 mt-1">
                Editing intensity level
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Processing Status */}
      <div className="space-y-4">
        <AnimatePresence mode="wait">
          {editingState.step === "analysis" &&
            editingState.analysisJob &&
            renderProgressCard(
              editingState.analysisJob,
              "🔍 Analyzing Video",
              "AI is detecting beats, motion, and scene changes..."
            )}

          {editingState.step === "editing" &&
            editingState.editingJob &&
            renderProgressCard(
              editingState.editingJob,
              "✨ Creating Your Edit",
              "AI is assembling your video with smart transitions..."
            )}

          {editingState.step === "completed" && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-2xl p-6 border border-green-200"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <CheckCircle className="w-8 h-8 text-green-500" />
                  <div>
                    <h3 className="font-semibold text-green-900">
                      Video Edit Complete!
                    </h3>
                    <p className="text-green-700 text-sm">
                      Your AI-edited video is ready for download
                    </p>
                  </div>
                </div>

                <div className="flex space-x-3">
                  <Button
                    variant="outline"
                    onClick={() =>
                      window.open(editingState.resultUrl, "_blank")
                    }
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Preview
                  </Button>
                  <Button
                    onClick={async () => {
                      try {
                        const response = await fetch(editingState.resultUrl!);
                        const data = await response.json();
                        if (data.download_url) {
                          window.open(data.download_url, "_blank");
                        }
                      } catch (error) {
                        console.error("Download failed:", error);
                      }
                    }}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>
              </div>
            </motion.div>
          )}

          {editingState.step === "error" && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-red-50 rounded-2xl p-6 border border-red-200"
            >
              <div className="flex items-center space-x-4">
                <AlertCircle className="w-8 h-8 text-red-500" />
                <div>
                  <h3 className="font-semibold text-red-900">
                    Processing Error
                  </h3>
                  <p className="text-red-700 text-sm">{editingState.error}</p>
                </div>
              </div>

              <div className="mt-4 flex space-x-3">
                <Button variant="outline" onClick={startAnalysis}>
                  Try Again
                </Button>
                <Button variant="ghost" onClick={onReset}>
                  Upload New Video
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
