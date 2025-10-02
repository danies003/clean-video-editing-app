"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Download,
  Settings,
  Zap,
  Wand2,
  CheckCircle,
  AlertCircle,
  Loader2,
  Palette,
  Sparkles,
  Layers,
  RotateCw,
  Target,
  Clock,
  Video,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  apiClient,
  VideoUploadResponse,
  ProcessingJob,
  AdvancedEditRequest,
  CompletedVideoResponse,
} from "@/lib/api";

interface EnhancedVideoEditorProps {
  uploadedVideo: VideoUploadResponse;
  onReset: () => void;
}

type EditingStep =
  | "analysis"
  | "editing"
  | "shader_composition"
  | "rendering"
  | "completed"
  | "error";

interface EditingState {
  step: EditingStep;
  analysisJob?: ProcessingJob;
  editingJob?: ProcessingJob;
  shaderJob?: ProcessingJob;
  renderJob?: ProcessingJob;
  error?: string;
  resultUrl?: string;
}

interface ShaderEffect {
  id: string;
  name: string;
  category: "transition" | "filter" | "distortion" | "color" | "motion";
  intensity: number;
  enabled: boolean;
  parameters: Record<string, unknown>;
}

interface RenderPreset {
  id: string;
  name: string;
  quality: "low" | "medium" | "high" | "ultra";
  description: string;
  estimatedTime: string;
}

const SHADER_EFFECTS: ShaderEffect[] = [
  {
    id: "glitch",
    name: "Glitch",
    category: "distortion",
    intensity: 0.5,
    enabled: false,
    parameters: { frequency: 0.1, intensity: 0.5 },
  },
  {
    id: "vintage",
    name: "Vintage",
    category: "color",
    intensity: 0.7,
    enabled: false,
    parameters: { saturation: 0.8, contrast: 1.2 },
  },
  {
    id: "neon",
    name: "Neon",
    category: "color",
    intensity: 0.6,
    enabled: false,
    parameters: { brightness: 1.3, saturation: 1.5 },
  },
  {
    id: "blur",
    name: "Motion Blur",
    category: "motion",
    intensity: 0.4,
    enabled: false,
    parameters: { radius: 5, direction: "horizontal" },
  },
  {
    id: "pixelate",
    name: "Pixelate",
    category: "distortion",
    intensity: 0.3,
    enabled: false,
    parameters: { blockSize: 8 },
  },
  {
    id: "wave",
    name: "Wave",
    category: "distortion",
    intensity: 0.5,
    enabled: false,
    parameters: { amplitude: 10, frequency: 2 },
  },
  {
    id: "zoom_blur",
    name: "Zoom Blur",
    category: "motion",
    intensity: 0.6,
    enabled: false,
    parameters: { center: [0.5, 0.5], strength: 0.8 },
  },
  {
    id: "chromatic",
    name: "Chromatic",
    category: "distortion",
    intensity: 0.4,
    enabled: false,
    parameters: { offset: 5 },
  },
];

const RENDER_PRESETS: RenderPreset[] = [
  {
    id: "fast",
    name: "Fast Preview",
    quality: "low",
    description: "Quick preview for testing",
    estimatedTime: "30s",
  },
  {
    id: "standard",
    name: "Standard",
    quality: "medium",
    description: "Balanced quality and speed",
    estimatedTime: "2m",
  },
  {
    id: "high",
    name: "High Quality",
    quality: "high",
    description: "Professional quality output",
    estimatedTime: "5m",
  },
  {
    id: "ultra",
    name: "Ultra HD",
    quality: "ultra",
    description: "Maximum quality for final output",
    estimatedTime: "10m",
  },
];

export function EnhancedVideoEditor({
  uploadedVideo,
  onReset,
}: EnhancedVideoEditorProps) {
  const [editingState, setEditingState] = useState<EditingState>({
    step: "analysis",
  });

  const [editSettings, setEditSettings] = useState<AdvancedEditRequest>({
    video_id: uploadedVideo.video_id,
    edit_scale: 1.0,
    style_preferences: {
      style: "tiktok",
      energy_level: "high",
      transition_style: "dynamic",
      pacing: "fast",
    },
    target_duration: undefined, // Let LLM determine optimal duration based on content
  });

  const [selectedShaderEffects, setSelectedShaderEffects] = useState<
    ShaderEffect[]
  >([]);
  const [selectedRenderPreset, setSelectedRenderPreset] =
    useState<RenderPreset>(RENDER_PRESETS[1]);
  const [showShaderPanel, setShowShaderPanel] = useState(false);
  const [showRenderPanel, setShowRenderPanel] = useState(false);
  const [autoTransitionEnabled, setAutoTransitionEnabled] = useState(true);
  const [crossVideoAnalysis, setCrossVideoAnalysis] = useState(false);

  // Poll job status with enhanced error handling
  const pollJobStatus = async (
    videoId: string,
    jobType: "analysis" | "editing" | "shader_composition" | "rendering"
  ) => {
    try {
      const response = await apiClient.getJobStatus(videoId);

      let job: ProcessingJob;
      let outputUrl: string | undefined = undefined;

      if ("job" in response && response.job) {
        job = response.job as ProcessingJob;
      } else if ("status" in response && response.status) {
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
        job = {
          job_id: "unknown",
          status: "failed",
          progress: 0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          error: "Invalid response format",
        };
      }

      // Update state based on job type
      setEditingState((prev) => {
        const updates: Partial<EditingState> = {};

        if (jobType === "analysis") {
          updates.analysisJob = job;
          if (job.status === "completed") {
            updates.step = "editing";
          } else if (job.status === "failed") {
            updates.step = "error";
            updates.error = job.error || "Analysis failed";
          }
        } else if (jobType === "editing") {
          updates.editingJob = job;
          if (job.status === "completed") {
            updates.step = "shader_composition";
          } else if (job.status === "failed") {
            updates.step = "error";
            updates.error = job.error || "Editing failed";
          }
        } else if (jobType === "shader_composition") {
          updates.shaderJob = job;
          if (job.status === "completed") {
            updates.step = "rendering";
          } else if (job.status === "failed") {
            updates.step = "error";
            updates.error = job.error || "Shader composition failed";
          }
        } else if (jobType === "rendering") {
          updates.renderJob = job;
          if (job.status === "completed") {
            updates.step = "completed";
            updates.resultUrl = outputUrl;
          } else if (job.status === "failed") {
            updates.step = "error";
            updates.error = job.error || "Rendering failed";
          }
        }

        return { ...prev, ...updates };
      });

      return job;
    } catch (error) {
      console.error(`Error polling ${jobType} job:`, error);
      setEditingState((prev) => ({
        ...prev,
        step: "error",
        error: `Failed to check ${jobType} status: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      }));
      return null;
    }
  };

  // Start analysis with enhanced settings
  const startAnalysis = async () => {
    try {
      setEditingState((prev) => ({ ...prev, step: "analysis" }));

      const response = await apiClient.analyzeVideo(uploadedVideo.video_id, {
        template_type: "beat_match",
        analysis_options: {
          edit_style: "tiktok",
          quality_preset: "high",
        },
        video_url: uploadedVideo.upload_url,
      });

      if (response.job?.job_id) {
        // Create a mock analysis job for now
        const mockAnalysisJob: ProcessingJob = {
          job_id: response.job.job_id,
          status: "processing",
          progress: 0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };

        setEditingState((prev) => ({ ...prev, analysisJob: mockAnalysisJob }));

        // Simulate progress for now
        const progressInterval = setInterval(() => {
          setEditingState((prev) => {
            if (prev.analysisJob && prev.analysisJob.progress < 100) {
              return {
                ...prev,
                analysisJob: {
                  ...prev.analysisJob!,
                  progress: prev.analysisJob!.progress + 20,
                  updated_at: new Date().toISOString(),
                },
              };
            } else {
              clearInterval(progressInterval);
              return {
                ...prev,
                step: "editing",
                analysisJob: prev.analysisJob
                  ? {
                      ...prev.analysisJob,
                      status: "completed",
                      progress: 100,
                      updated_at: new Date().toISOString(),
                    }
                  : prev.analysisJob,
              };
            }
          });
        }, 1000);
      }
    } catch (error) {
      console.error("Error starting analysis:", error);
      setEditingState((prev) => ({
        ...prev,
        step: "error",
        error: `Analysis failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      }));
    }
  };

  // Start enhanced editing with shader composition
  const startEditing = async () => {
    try {
      setEditingState((prev) => ({ ...prev, step: "editing" }));

      const enhancedEditRequest: AdvancedEditRequest = {
        video_id: uploadedVideo.video_id,
        edit_scale: 0.8,
        style_preferences: {
          style: editSettings.style_preferences?.style || "tiktok",
          energy_level: "high",
          transition_style: "dynamic",
          pacing: "fast",
        },
        target_duration: editSettings.target_duration,
      };

      const response = await apiClient.advancedEdit(enhancedEditRequest);

      if (response.job?.job_id) {
        // Create a mock editing job
        const mockEditingJob: ProcessingJob = {
          job_id: response.job.job_id,
          status: "processing",
          progress: 0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };

        setEditingState((prev) => ({ ...prev, editingJob: mockEditingJob }));

        // Simulate progress for now
        const progressInterval = setInterval(() => {
          setEditingState((prev) => {
            if (prev.editingJob && prev.editingJob.progress < 100) {
              return {
                ...prev,
                editingJob: {
                  ...prev.editingJob!,
                  progress: prev.editingJob!.progress + 25,
                  updated_at: new Date().toISOString(),
                },
              };
            } else {
              clearInterval(progressInterval);
              return {
                ...prev,
                step: "shader_composition",
                editingJob: prev.editingJob
                  ? {
                      ...prev.editingJob,
                      status: "completed",
                      progress: 100,
                      updated_at: new Date().toISOString(),
                    }
                  : prev.editingJob,
              };
            }
          });
        }, 800);
      }
    } catch (error) {
      console.error("Error starting editing:", error);
      setEditingState((prev) => ({
        ...prev,
        step: "error",
        error: `Editing failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      }));
    }
  };

  // Start shader composition
  const startShaderComposition = async () => {
    try {
      setEditingState((prev) => ({ ...prev, step: "shader_composition" }));

      // For now, we'll simulate shader composition since the backend doesn't have this endpoint yet
      // In a real implementation, you'd call the shader composition API
      console.log(
        "Starting shader composition with effects:",
        selectedShaderEffects.filter((effect) => effect.enabled)
      );

      // Simulate shader composition job
      const mockShaderJob: ProcessingJob = {
        job_id: `shader_${Date.now()}`,
        status: "processing",
        progress: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      setEditingState((prev) => ({ ...prev, shaderJob: mockShaderJob }));

      // Simulate progress
      const progressInterval = setInterval(() => {
        setEditingState((prev) => {
          if (prev.shaderJob && prev.shaderJob.progress < 100) {
            return {
              ...prev,
              shaderJob: {
                ...prev.shaderJob!,
                progress: prev.shaderJob!.progress + 10,
                updated_at: new Date().toISOString(),
              },
            };
          } else {
            clearInterval(progressInterval);
            return {
              ...prev,
              step: "rendering",
              shaderJob: prev.shaderJob
                ? {
                    ...prev.shaderJob,
                    status: "completed",
                    progress: 100,
                    updated_at: new Date().toISOString(),
                  }
                : prev.shaderJob,
            };
          }
        });
      }, 500);
    } catch (error) {
      console.error("Error starting shader composition:", error);
      setEditingState((prev) => ({
        ...prev,
        step: "error",
        error: `Shader composition failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      }));
    }
  };

  // Start rendering with enhanced presets
  const startRendering = async () => {
    try {
      setEditingState((prev) => ({ ...prev, step: "rendering" }));

      // Use the existing advanced edit endpoint for rendering
      const renderRequest: AdvancedEditRequest = {
        video_id: uploadedVideo.video_id,
        edit_scale: 0.8,
        style_preferences: {
          style: editSettings.style_preferences?.style || "tiktok",
          energy_level: "high",
          transition_style: "dynamic",
          pacing: "fast",
        },
        target_duration: editSettings.target_duration,
      };

      const response = await apiClient.advancedEdit(renderRequest);

      if (response.job?.job_id) {
        // Create a mock render job
        const mockRenderJob: ProcessingJob = {
          job_id: response.job.job_id,
          status: "processing",
          progress: 0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };

        setEditingState((prev) => ({ ...prev, renderJob: mockRenderJob }));

        // Simulate progress for now
        const progressInterval = setInterval(() => {
          setEditingState((prev) => {
            if (prev.renderJob && prev.renderJob.progress < 100) {
              return {
                ...prev,
                renderJob: {
                  ...prev.renderJob!,
                  progress: prev.renderJob!.progress + 15,
                  updated_at: new Date().toISOString(),
                },
              };
            } else {
              clearInterval(progressInterval);
              return {
                ...prev,
                step: "completed",
                renderJob: prev.renderJob
                  ? {
                      ...prev.renderJob,
                      status: "completed",
                      progress: 100,
                      result: {
                        output_url: "https://example.com/completed-video.mp4", // Mock URL
                      },
                    }
                  : prev.renderJob,
                resultUrl: "https://example.com/completed-video.mp4", // Mock URL
              };
            }
          });
        }, 1200);
      }
    } catch (error) {
      console.error("Error starting rendering:", error);
      setEditingState((prev) => ({
        ...prev,
        step: "error",
        error: `Rendering failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      }));
    }
  };

  // Enhanced progress card with better visual feedback
  const renderProgressCard = (
    job: ProcessingJob,
    title: string,
    description: string,
    icon: React.ReactNode,
    onAction?: () => void,
    actionText?: string
  ) => {
    const getStatusColor = () => {
      switch (job.status) {
        case "completed":
          return "text-green-500";
        case "failed":
          return "text-red-500";
        case "processing":
          return "text-blue-500";
        default:
          return "text-gray-500";
      }
    };

    const getStatusIcon = () => {
      switch (job.status) {
        case "completed":
          return <CheckCircle className="w-5 h-5 text-green-500" />;
        case "failed":
          return <AlertCircle className="w-5 h-5 text-red-500" />;
        case "processing":
          return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
        default:
          return <Clock className="w-5 h-5 text-gray-500" />;
      }
    };

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card className="w-full">
          <CardHeader>
            <div className="flex items-center gap-3">
              {icon}
              <div className="flex-1">
                <CardTitle className="text-lg">{title}</CardTitle>
                <CardDescription>{description}</CardDescription>
              </div>
              {getStatusIcon()}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className={getStatusColor()}>
                  {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                </span>
                <span className="text-gray-500">{job.progress}%</span>
              </div>

              <div className="w-full bg-gray-200 rounded-full h-2">
                <motion.div
                  className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${job.progress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>

              {job.error && (
                <div className="text-red-500 text-sm bg-red-50 p-2 rounded">
                  {job.error}
                </div>
              )}

              {onAction && actionText && (
                <Button
                  onClick={onAction}
                  disabled={job.status === "processing"}
                  className="w-full"
                >
                  {actionText}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  };

  // Shader effects panel
  const renderShaderPanel = () => (
    <motion.div
      initial={{ opacity: 0, x: 300 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 300 }}
      className="fixed right-0 top-0 h-full w-80 bg-white shadow-xl border-l p-6 overflow-y-auto"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold">Shader Effects</h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowShaderPanel(false)}
        >
          <X className="w-4 h-4" />
        </Button>
      </div>

      <div className="space-y-4">
        {SHADER_EFFECTS.map((effect) => (
          <Card key={effect.id} className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={effect.enabled}
                  onChange={(e) => {
                    const updatedEffects = selectedShaderEffects.map((ef) =>
                      ef.id === effect.id
                        ? { ...ef, enabled: e.target.checked }
                        : ef
                    );
                    setSelectedShaderEffects(updatedEffects);
                  }}
                  className="rounded"
                />
                <span className="font-medium">{effect.name}</span>
              </div>
              <span className="text-xs px-2 py-1 bg-gray-100 rounded">
                {effect.category}
              </span>
            </div>

            {effect.enabled && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-600">Intensity:</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={effect.intensity}
                    onChange={(e) => {
                      const updatedEffects = selectedShaderEffects.map((ef) =>
                        ef.id === effect.id
                          ? { ...ef, intensity: parseFloat(e.target.value) }
                          : ef
                      );
                      setSelectedShaderEffects(updatedEffects);
                    }}
                    className="flex-1"
                  />
                  <span className="text-sm text-gray-500 w-8">
                    {Math.round(effect.intensity * 100)}%
                  </span>
                </div>
              </div>
            )}
          </Card>
        ))}
      </div>
    </motion.div>
  );

  // Render presets panel
  const renderRenderPanel = () => (
    <motion.div
      initial={{ opacity: 0, x: 300 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 300 }}
      className="fixed right-0 top-0 h-full w-80 bg-white shadow-xl border-l p-6 overflow-y-auto"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold">Render Settings</h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowRenderPanel(false)}
        >
          <X className="w-4 h-4" />
        </Button>
      </div>

      <div className="space-y-4">
        {RENDER_PRESETS.map((preset) => (
          <Card
            key={preset.id}
            className={`p-4 cursor-pointer transition-all ${
              selectedRenderPreset.id === preset.id
                ? "ring-2 ring-blue-500 bg-blue-50"
                : "hover:bg-gray-50"
            }`}
            onClick={() => setSelectedRenderPreset(preset)}
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium">{preset.name}</h4>
              <span className="text-xs px-2 py-1 bg-gray-100 rounded">
                {preset.quality}
              </span>
            </div>
            <p className="text-sm text-gray-600 mb-2">{preset.description}</p>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Clock className="w-3 h-3" />
              <span>~{preset.estimatedTime}</span>
            </div>
          </Card>
        ))}
      </div>
    </motion.div>
  );

  // Main render method
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-3xl font-bold text-white mb-2">
            Enhanced Video Editor
          </h1>
          <p className="text-gray-300">
            AI-powered editing with advanced shader effects and rendering
          </p>
        </motion.div>

        {/* Control Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          <Button
            onClick={() => setShowShaderPanel(true)}
            className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
          >
            <Palette className="w-4 h-4 mr-2" />
            Shader Effects
          </Button>

          <Button
            onClick={() => setShowRenderPanel(true)}
            className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
          >
            <Settings className="w-4 h-4 mr-2" />
            Render Settings
          </Button>

          <Button
            onClick={() => setAutoTransitionEnabled(!autoTransitionEnabled)}
            className={`${
              autoTransitionEnabled
                ? "bg-gradient-to-r from-green-600 to-emerald-600"
                : "bg-gradient-to-r from-gray-600 to-gray-700"
            }`}
          >
            <Sparkles className="w-4 h-4 mr-2" />
            Auto Transitions
          </Button>

          <Button
            onClick={() => setCrossVideoAnalysis(!crossVideoAnalysis)}
            className={`${
              crossVideoAnalysis
                ? "bg-gradient-to-r from-orange-600 to-red-600"
                : "bg-gradient-to-r from-gray-600 to-gray-700"
            }`}
          >
            <Layers className="w-4 h-4 mr-2" />
            Cross Analysis
          </Button>
        </motion.div>

        {/* Progress Steps */}
        <div className="space-y-4">
          {editingState.step === "analysis" && (
            <div className="space-y-4">
              {editingState.analysisJob ? (
                renderProgressCard(
                  editingState.analysisJob,
                  "Video Analysis",
                  "Analyzing video content, beats, and motion patterns",
                  <Target className="w-5 h-5 text-blue-500" />
                )
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="w-5 h-5 text-blue-500" />
                      Start Analysis
                    </CardTitle>
                    <CardDescription>
                      Begin AI-powered video analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button onClick={startAnalysis} className="w-full">
                      <Zap className="w-4 h-4 mr-2" />
                      Start Analysis
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {editingState.step === "editing" && (
            <div className="space-y-4">
              {editingState.editingJob ? (
                renderProgressCard(
                  editingState.editingJob,
                  "AI Editing",
                  "Applying intelligent cuts and transitions",
                  <Wand2 className="w-5 h-5 text-purple-500" />
                )
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Wand2 className="w-5 h-5 text-purple-500" />
                      Start Editing
                    </CardTitle>
                    <CardDescription>
                      Apply AI-powered editing with selected effects
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button onClick={startEditing} className="w-full">
                      <Wand2 className="w-4 h-4 mr-2" />
                      Start Editing
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {editingState.step === "shader_composition" && (
            <div className="space-y-4">
              {editingState.shaderJob ? (
                renderProgressCard(
                  editingState.shaderJob,
                  "Shader Composition",
                  "Composing advanced visual effects",
                  <Palette className="w-5 h-5 text-pink-500" />
                )
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Palette className="w-5 h-5 text-pink-500" />
                      Start Shader Composition
                    </CardTitle>
                    <CardDescription>
                      Apply selected shader effects and transitions
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button onClick={startShaderComposition} className="w-full">
                      <Palette className="w-4 h-4 mr-2" />
                      Start Shader Composition
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {editingState.step === "rendering" && (
            <div className="space-y-4">
              {editingState.renderJob ? (
                renderProgressCard(
                  editingState.renderJob,
                  "Video Rendering",
                  `Rendering with ${selectedRenderPreset.name} preset`,
                  <Video className="w-5 h-5 text-green-500" />
                )
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Video className="w-5 h-5 text-green-500" />
                      Start Rendering
                    </CardTitle>
                    <CardDescription>
                      Render final video with selected quality preset
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button onClick={startRendering} className="w-full">
                      <Video className="w-4 h-4 mr-2" />
                      Start Rendering
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {editingState.step === "completed" && editingState.resultUrl && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center"
            >
              <Card className="p-8">
                <div className="flex flex-col items-center gap-4">
                  <CheckCircle className="w-16 h-16 text-green-500" />
                  <h3 className="text-2xl font-bold">Video Completed!</h3>
                  <p className="text-gray-600">
                    Your enhanced video has been successfully rendered
                  </p>
                  <div className="flex gap-4">
                    <Button
                      onClick={() =>
                        window.open(editingState.resultUrl, "_blank")
                      }
                      className="bg-gradient-to-r from-green-600 to-emerald-600"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download Video
                    </Button>
                    <Button onClick={onReset} variant="outline">
                      <RotateCw className="w-4 h-4 mr-2" />
                      Start Over
                    </Button>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          {editingState.step === "error" && editingState.error && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <Card className="p-6 border-red-200 bg-red-50">
                <div className="flex items-center gap-3 mb-4">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                  <h3 className="text-lg font-semibold text-red-800">Error</h3>
                </div>
                <p className="text-red-700 mb-4">{editingState.error}</p>
                <Button
                  onClick={onReset}
                  variant="outline"
                  className="border-red-300 text-red-700 hover:bg-red-100"
                >
                  <RotateCw className="w-4 h-4 mr-2" />
                  Try Again
                </Button>
              </Card>
            </motion.div>
          )}
        </div>
      </div>

      {/* Shader Panel */}
      <AnimatePresence>
        {showShaderPanel && renderShaderPanel()}
      </AnimatePresence>

      {/* Render Panel */}
      <AnimatePresence>
        {showRenderPanel && renderRenderPanel()}
      </AnimatePresence>
    </div>
  );
}
