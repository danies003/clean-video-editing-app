"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Video,
  SkipBack,
  SkipForward,
  AlertTriangle,
  Upload,
  FileVideo,
  X,
  CheckCircle,
  Settings,
  Sparkles,
  Eye,
} from "lucide-react";
import { SegmentEditor } from "./SegmentEditor";
import { TransitionEditor } from "./TransitionEditor";

interface EditDecisionSegment {
  start: number;
  end: number;
  transition?: string;
  transition_duration?: number;
  tags: string[];
  speed?: number;
  transition_in?: string;
  transition_out?: string;
  enabled?: boolean;
  effectCustomizations?: {
    [effectName: string]: {
      enabled: boolean;
      parameters: { [paramName: string]: any };
    };
  };
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

interface VideoTimelineEditorProps {
  videoFile?: File;
  convertedVideoUrl?: string;
  segments: EditDecisionSegment[];
  onSegmentUpdate: (segmentIndex: number, segment: EditDecisionSegment) => void;
  onPlaybackTimeChange: (time: number) => void;
  llmSuggestions?: LLMSuggestion[];
  onBack?: () => void;
  onProceed?: () => void;
  customDuration?: number; // For multi-video projects
  projectId?: string; // Add project ID for LLM testing
}

const VideoTimelineEditor: React.FC<VideoTimelineEditorProps> = ({
  videoFile,
  convertedVideoUrl,
  segments,
  onSegmentUpdate,
  onPlaybackTimeChange,
  llmSuggestions = [],
  onBack,
  onProceed,
  customDuration,
  projectId,
}) => {
  // Debug logging
  console.log("🔧 [VideoTimelineEditor] segments length:", segments.length);
  console.log("🔧 [VideoTimelineEditor] segments:", segments);
  console.log("🔧 [VideoTimelineEditor] customDuration:", customDuration);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [showEffects, setShowEffects] = useState(true);
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(-1);
  const [effectApplied, setEffectApplied] = useState(false);
  const [activeEffects, setActiveEffects] = useState<string[]>([]);
  const [videoOrientation, setVideoOrientation] = useState<
    "horizontal" | "vertical"
  >("horizontal");
  const [isProcessingPlayToggle, setIsProcessingPlayToggle] =
    useState<boolean>(false);

  // LLM timeline testing state
  const [isTestingLLM, setIsTestingLLM] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState<any>(null);
  const [showLLMResults, setShowLLMResults] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const [isSegmentEditorOpen, setIsSegmentEditorOpen] = useState(false);
  const [editingSegment, setEditingSegment] =
    useState<EditDecisionSegment | null>(null);
  const [editingSegmentIndex, setEditingSegmentIndex] = useState(-1);
  const [editingTransitionIndex, setEditingTransitionIndex] =
    useState<number>(-1);
  const [editingTransition, setEditingTransition] = useState<any>(null);
  const [showTransitionEditor, setShowTransitionEditor] = useState(false);

  const originalVideoRef = useRef<HTMLVideoElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    // If we have a video file, create object URL from it
    if (videoFile && videoFile instanceof File) {
      console.log("🎬 VideoTimelineEditor: Creating object URL from file:", {
        name: videoFile.name,
        size: videoFile.size,
      });
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    // If no video file but we have a converted URL, use that
    else if (convertedVideoUrl) {
      console.log(
        "🎬 VideoTimelineEditor: Using converted video URL:",
        convertedVideoUrl
      );
      setVideoUrl(convertedVideoUrl);
    } else {
      console.log("🎬 VideoTimelineEditor: No valid video file or URL", {
        videoFile,
        convertedVideoUrl,
      });
    }
  }, [videoFile, convertedVideoUrl]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getSegmentAtTime = (time: number) => {
    return segments.findIndex(
      (segment) => time >= segment.start && time <= segment.end
    );
  };

  // Track last video time for seeking detection
  const [lastVideoTime, setLastVideoTime] = useState<number>(0);

  // Simple normalized time calculation for playhead positioning
  const getNormalizedPlayheadTime = () => {
    if (!originalVideoRef.current) return currentTime;
    const video = originalVideoRef.current;
    const playbackRate = video.playbackRate;

    // If speed is normal, use currentTime directly
    if (playbackRate === 1.0) {
      return currentTime;
    }

    // For speed changes, calculate the "real time" position
    // When video plays at 2x, currentTime advances faster, so we divide by rate
    return currentTime / playbackRate;
  };

  // Complex real-time effects application with full functionality
  const applyRealTimeEffects = useCallback(
    (currentVideoTime?: number) => {
      if (!canvasRef.current || !originalVideoRef.current) {
        console.log("[DEBUG] Missing canvas or video ref");
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      const video = originalVideoRef.current;

      if (!ctx || !video) {
        console.log("[DEBUG] Missing context or video");
        return;
      }

      // Check if video is ready
      if (video.readyState < 2) {
        console.log("[DEBUG] Video not ready, readyState:", video.readyState);
        return;
      }

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Clear the canvas first
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw current video frame with error handling for CORS issues
      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        console.log(
          "[DEBUG] Drew frame at time:",
          video.currentTime,
          "canvas size:",
          canvas.width,
          "x",
          canvas.height
        );
      } catch (error) {
        console.warn("[DEBUG] CORS error when drawing video to canvas:", error);
        // Draw a placeholder instead
        ctx.fillStyle = "#1f2937";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#6b7280";
        ctx.font = "16px Arial";
        ctx.textAlign = "center";
        ctx.fillText("Video Preview", canvas.width / 2, canvas.height / 2);
        return; // Skip effects if we can't draw the video
      }

      // Get current segment using video's current time directly
      const videoTime =
        currentVideoTime !== undefined ? currentVideoTime : video.currentTime;

      // Use videoTime directly for segment detection (no normalized time)
      const currentSegment = segments.find(
        (segment) => videoTime >= segment.start && videoTime < segment.end
      );

      // Check if we're in a transition period
      const nextSegment = segments.find(
        (segment) =>
          segment.start > videoTime && segment.start <= videoTime + 0.5
      );
      const isNearSegmentBoundary =
        nextSegment && nextSegment.start - videoTime <= 0.5;
      const isInTransitionPeriod =
        isNearSegmentBoundary && nextSegment && currentSegment;

      console.log(
        "[DEBUG] Current time:",
        videoTime.toFixed(2),
        "Found segment:",
        currentSegment
          ? `start=${currentSegment.start}, end=${
              currentSegment.end
            }, tags=${currentSegment.tags.join(",")}`
          : "none",
        "In transition period:",
        isInTransitionPeriod
      );

      if (currentSegment && currentSegment.tags.length > 0) {
        console.log("[DEBUG] Found segment:", currentSegment);
        console.log("[DEBUG] Segment enabled:", currentSegment.enabled);
        console.log(
          "[DEBUG] Effect customizations:",
          currentSegment.effectCustomizations
        );

        // Check if segment is enabled (default to true if not specified)
        const isSegmentEnabled = currentSegment.enabled !== false;

        if (!isSegmentEnabled) {
          console.log("[DEBUG] Segment is disabled, not applying effects");
          setActiveEffects([]);
          return;
        }

        // Filter enabled effects (exclude speed effects as they're handled separately)
        const enabledEffects = currentSegment.tags.filter((tag) => {
          // Skip speed effects in visual effects processing
          if (tag === "speed_up" || tag === "slow_motion" || tag === "speed") {
            return false;
          }

          const effectCustomization =
            currentSegment.effectCustomizations?.[tag];
          const isEffectEnabled = effectCustomization?.enabled !== false; // Default to true if not specified
          console.log(`[DEBUG] Effect ${tag} enabled: ${isEffectEnabled}`);
          return isEffectEnabled;
        });

        console.log("[DEBUG] Enabled visual effects:", enabledEffects);
        setActiveEffects(enabledEffects);

        // COORDINATE VISUAL EFFECTS WITH SPEED CHANGES
        // During transitions, reduce the intensity of visual effects to prevent conflicts
        const speedFactor = video.playbackRate;
        const transitionIntensity = isInTransitionPeriod ? 0.5 : 1.0; // Reduce intensity during transitions

        // Apply only enabled effects in sequence (speed effects handled separately above)
        enabledEffects.forEach((tag) => {
          console.log(
            `[DEBUG] Applying visual effect: ${tag} with intensity: ${transitionIntensity}`
          );
          switch (tag) {
            case "brightness":
              applyBrightnessEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.2 * transitionIntensity
              );
              break;
            case "contrast":
              applyContrastEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.3 * transitionIntensity
              );
              break;
            case "saturation":
              applySaturationEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.2 * transitionIntensity
              );
              break;
            case "blur":
              applyBlurEffect(
                ctx,
                canvas.width,
                canvas.height,
                2 * transitionIntensity
              );
              break;
            case "vintage":
              applyVintageEffect(ctx, canvas.width, canvas.height);
              break;
            case "black_white":
              applyBlackWhiteEffect(ctx, canvas.width, canvas.height);
              break;
            case "warm":
              applyBrightnessEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.1 * transitionIntensity
              );
              applySaturationEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.2 * transitionIntensity
              );
              break;
            case "cinematic_look":
              applyContrastEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.3 * transitionIntensity
              );
              applyVintageEffect(ctx, canvas.width, canvas.height);
              break;
            case "dramatic_lighting":
              applyContrastEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.4 * transitionIntensity
              );
              break;
            // Enhanced effects from the enhanced plan
            case "motion_blur":
              applyBlurEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.5 * transitionIntensity
              );
              break;
            case "cinematic":
              applyContrastEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.3 * transitionIntensity
              );
              applyVintageEffect(ctx, canvas.width, canvas.height);
              break;
            case "color_grading":
              applySaturationEffect(
                ctx,
                canvas.width,
                canvas.height,
                1.3 * transitionIntensity
              );
              break;
            case "beat_sync":
              // Visual beat sync effect
              const beatIntensity = Math.sin(videoTime * 10) * 0.1 + 1.0;
              if (ctx) {
                applyBrightnessEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  beatIntensity * transitionIntensity
                );
              }
              break;
            case "volume_wave":
              // Volume wave visualization effect
              const waveIntensity = Math.sin(videoTime * 8) * 0.15 + 1.0;
              if (ctx) {
                applySaturationEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  waveIntensity * transitionIntensity
                );
              }
              break;
            case "audio_pulse":
              // Audio pulse effect
              const pulseIntensity = Math.sin(videoTime * 12) * 0.2 + 1.0;
              if (ctx) {
                applyBrightnessEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  pulseIntensity * transitionIntensity
                );
              }
              break;
            case "frequency_visualizer":
              // Frequency visualization effect
              const freqIntensity = Math.sin(videoTime * 6) * 0.25 + 1.0;
              if (ctx) {
                applyContrastEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  freqIntensity * transitionIntensity
                );
              }
              break;
            case "cyberpunk":
              // Cyberpunk effect
              if (ctx) {
                applySaturationEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.5 * transitionIntensity
                );
                applyContrastEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.4 * transitionIntensity
                );
              }
              break;
            case "glitch":
              // Glitch effect - random color shifts
              const glitchIntensity = Math.sin(videoTime * 20) * 0.3;
              if (Math.abs(glitchIntensity) > 0.2 && ctx) {
                applySaturationEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.0 + glitchIntensity * transitionIntensity
                );
              }
              break;
            case "optical_flow":
              // Optical flow effect
              if (ctx) {
                applyBlurEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.2 * transitionIntensity
                );
              }
              break;
            case "motion_trail":
              // Motion trail effect
              if (ctx) {
                applyBlurEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.8 * transitionIntensity
                );
              }
              break;
            case "scene_transition":
              // Scene transition effect
              if (ctx) {
                applyContrastEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.3 * transitionIntensity
                );
              }
              break;
            case "cartoon":
              // Cartoon effect
              if (ctx) {
                applySaturationEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.4 * transitionIntensity
                );
                applyContrastEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.6 * transitionIntensity
                );
              }
              break;
            case "fisheye":
              // Fisheye effect - barrel distortion
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const maxDistance = Math.sqrt(
                  centerX * centerX + centerY * centerY
                );

                for (let y = 0; y < canvas.height; y++) {
                  for (let x = 0; x < canvas.width; x++) {
                    const dx = x - centerX;
                    const dy = y - centerY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const normalizedDistance = distance / maxDistance;
                    const distortion =
                      normalizedDistance * normalizedDistance * 0.3;
                    const newDistance = distance * (1 + distortion);
                    const angle = Math.atan2(dy, dx);
                    const newX = centerX + newDistance * Math.cos(angle);
                    const newY = centerY + newDistance * Math.sin(angle);

                    if (
                      newX >= 0 &&
                      newX < canvas.width &&
                      newY >= 0 &&
                      newY < canvas.height
                    ) {
                      const srcIndex = (y * canvas.width + x) * 4;
                      const dstIndex =
                        (Math.floor(newY) * canvas.width + Math.floor(newX)) *
                        4;
                      data[dstIndex] = data[srcIndex];
                      data[dstIndex + 1] = data[srcIndex + 1];
                      data[dstIndex + 2] = data[srcIndex + 2];
                      data[dstIndex + 3] = data[srcIndex + 3];
                    }
                  }
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "warp":
              // Warp effect - wave distortion
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;

                for (let y = 0; y < canvas.height; y++) {
                  for (let x = 0; x < canvas.width; x++) {
                    const waveX = x + Math.sin(y * 0.02 + videoTime * 3) * 10;
                    const waveY = y + Math.cos(x * 0.02 + videoTime * 2) * 8;

                    if (
                      waveX >= 0 &&
                      waveX < canvas.width &&
                      waveY >= 0 &&
                      waveY < canvas.height
                    ) {
                      const srcIndex = (y * canvas.width + x) * 4;
                      const dstIndex =
                        (Math.floor(waveY) * canvas.width + Math.floor(waveX)) *
                        4;
                      data[dstIndex] = data[srcIndex];
                      data[dstIndex + 1] = data[srcIndex + 1];
                      data[dstIndex + 2] = data[srcIndex + 2];
                      data[dstIndex + 3] = data[srcIndex + 3];
                    }
                  }
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "perspective":
              // Perspective effect - tilt shift
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;
                const centerY = canvas.height / 2;

                for (let y = 0; y < canvas.height; y++) {
                  for (let x = 0; x < canvas.width; x++) {
                    const distanceFromCenter = Math.abs(y - centerY) / centerY;
                    const shift =
                      distanceFromCenter * Math.sin(videoTime * 2) * 20;
                    const newX = x + shift;

                    if (
                      newX >= 0 &&
                      newX < canvas.width &&
                      y >= 0 &&
                      y < canvas.height
                    ) {
                      const srcIndex = (y * canvas.width + x) * 4;
                      const dstIndex =
                        (y * canvas.width + Math.floor(newX)) * 4;
                      data[dstIndex] = data[srcIndex];
                      data[dstIndex + 1] = data[srcIndex + 1];
                      data[dstIndex + 2] = data[srcIndex + 2];
                      data[dstIndex + 3] = data[srcIndex + 3];
                    }
                  }
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "invert":
              // Invert colors effect
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;

                for (let i = 0; i < data.length; i += 4) {
                  data[i] = 255 - data[i]; // Red
                  data[i + 1] = 255 - data[i + 1]; // Green
                  data[i + 2] = 255 - data[i + 2]; // Blue
                }

                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "high_contrast":
              if (ctx) {
                applyContrastEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.5 * transitionIntensity
                );
              }
              break;
            case "film_noir":
              if (ctx) {
                applyBlackWhiteEffect(ctx, canvas.width, canvas.height);
                applyContrastEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.4 * transitionIntensity
                );
              }
              break;
            case "duotone":
              // Duotone effect
              if (ctx) {
                applySaturationEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  0.8 * transitionIntensity
                );
                applyContrastEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  1.3 * transitionIntensity
                );
              }
              break;
            case "twirl":
              // Twirl effect - create a spiral distortion
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const maxDistance = Math.sqrt(
                  centerX * centerX + centerY * centerY
                );

                for (let y = 0; y < canvas.height; y++) {
                  for (let x = 0; x < canvas.width; x++) {
                    const dx = x - centerX;
                    const dy = y - centerY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const angle =
                      Math.atan2(dy, dx) +
                      (distance / maxDistance) * Math.sin(videoTime * 2) * 0.5;
                    const newX = centerX + distance * Math.cos(angle);
                    const newY = centerY + distance * Math.sin(angle);

                    if (
                      newX >= 0 &&
                      newX < canvas.width &&
                      newY >= 0 &&
                      newY < canvas.height
                    ) {
                      const srcIndex = (y * canvas.width + x) * 4;
                      const dstIndex =
                        (Math.floor(newY) * canvas.width + Math.floor(newX)) *
                        4;
                      data[dstIndex] = data[srcIndex];
                      data[dstIndex + 1] = data[srcIndex + 1];
                      data[dstIndex + 2] = data[srcIndex + 2];
                      data[dstIndex + 3] = data[srcIndex + 3];
                    }
                  }
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "vintage":
              // Vintage effect - sepia tone
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;

                for (let i = 0; i < data.length; i += 4) {
                  const r = data[i];
                  const g = data[i + 1];
                  const b = data[i + 2];

                  // Sepia conversion
                  const tr = 0.393 * r + 0.769 * g + 0.189 * b;
                  const tg = 0.349 * r + 0.686 * g + 0.168 * b;
                  const tb = 0.272 * r + 0.534 * g + 0.131 * b;

                  data[i] = Math.min(255, tr);
                  data[i + 1] = Math.min(255, tg);
                  data[i + 2] = Math.min(255, tb);
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "cross_dissolve":
              // Cross dissolve effect - fade transition
              if (ctx) {
                const fadeIntensity = Math.sin(videoTime * 2) * 0.2 + 0.8;
                applyBrightnessEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  fadeIntensity * transitionIntensity
                );
              }
              break;
            case "slide":
              // Slide effect - horizontal movement
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;
                const slideOffset = Math.sin(videoTime * 3) * 20;

                for (let y = 0; y < canvas.height; y++) {
                  for (let x = 0; x < canvas.width; x++) {
                    const newX = x + slideOffset;
                    if (newX >= 0 && newX < canvas.width) {
                      const srcIndex = (y * canvas.width + x) * 4;
                      const dstIndex =
                        (y * canvas.width + Math.floor(newX)) * 4;
                      data[dstIndex] = data[srcIndex];
                      data[dstIndex + 1] = data[srcIndex + 1];
                      data[dstIndex + 2] = data[srcIndex + 2];
                      data[dstIndex + 3] = data[srcIndex + 3];
                    }
                  }
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "zoom":
              // Zoom effect - scale transformation
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const zoomFactor = 1.0 + Math.sin(videoTime * 2) * 0.2;

                for (let y = 0; y < canvas.height; y++) {
                  for (let x = 0; x < canvas.width; x++) {
                    const dx = (x - centerX) / zoomFactor;
                    const dy = (y - centerY) / zoomFactor;
                    const newX = centerX + dx;
                    const newY = centerY + dy;

                    if (
                      newX >= 0 &&
                      newX < canvas.width &&
                      newY >= 0 &&
                      newY < canvas.height
                    ) {
                      const srcIndex = (y * canvas.width + x) * 4;
                      const dstIndex =
                        (Math.floor(newY) * canvas.width + Math.floor(newX)) *
                        4;
                      data[dstIndex] = data[srcIndex];
                      data[dstIndex + 1] = data[srcIndex + 1];
                      data[dstIndex + 2] = data[srcIndex + 2];
                      data[dstIndex + 3] = data[srcIndex + 3];
                    }
                  }
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "whip_pan":
              // Whip pan effect - fast horizontal blur
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;
                const panSpeed = Math.sin(videoTime * 5) * 15;

                for (let y = 0; y < canvas.height; y++) {
                  for (let x = 0; x < canvas.width; x++) {
                    const newX = x + panSpeed;
                    if (newX >= 0 && newX < canvas.width) {
                      const srcIndex = (y * canvas.width + x) * 4;
                      const dstIndex =
                        (y * canvas.width + Math.floor(newX)) * 4;
                      data[dstIndex] = data[srcIndex];
                      data[dstIndex + 1] = data[srcIndex + 1];
                      data[dstIndex + 2] = data[srcIndex + 2];
                      data[dstIndex + 3] = data[srcIndex + 3];
                    }
                  }
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            case "spin":
              // Spin effect - rotation transformation
              if (ctx) {
                const imageData = ctx.getImageData(
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );
                const data = imageData.data;
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const rotationAngle = videoTime * 2; // Rotate over time

                for (let y = 0; y < canvas.height; y++) {
                  for (let x = 0; x < canvas.width; x++) {
                    const dx = x - centerX;
                    const dy = y - centerY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const angle = Math.atan2(dy, dx) + rotationAngle;
                    const newX = centerX + distance * Math.cos(angle);
                    const newY = centerY + distance * Math.sin(angle);

                    if (
                      newX >= 0 &&
                      newX < canvas.width &&
                      newY >= 0 &&
                      newY < canvas.height
                    ) {
                      const srcIndex = (y * canvas.width + x) * 4;
                      const dstIndex =
                        (Math.floor(newY) * canvas.width + Math.floor(newX)) *
                        4;
                      data[dstIndex] = data[srcIndex];
                      data[dstIndex + 1] = data[srcIndex + 1];
                      data[dstIndex + 2] = data[srcIndex + 2];
                      data[dstIndex + 3] = data[srcIndex + 3];
                    }
                  }
                }
                ctx.putImageData(imageData, 0, 0);
              }
              break;
            default:
              console.log(`[DEBUG] Unknown effect: ${tag}`);
              break;
          }
        });

        // Apply transitions if we're near a segment boundary
        const currentSegmentIndex = segments.findIndex(
          (segment) => videoTime >= segment.start && videoTime <= segment.end
        );

        if (currentSegmentIndex >= 0) {
          const currentSegment = segments[currentSegmentIndex];
          const nextSegment = segments[currentSegmentIndex + 1];

          // Check if we're in a transition period (last 0.5 seconds of current segment)
          const timeUntilEnd = currentSegment.end - videoTime;
          const transitionDuration = currentSegment.transition_duration || 0.5;

          if (
            timeUntilEnd <= transitionDuration &&
            timeUntilEnd > 0 &&
            nextSegment
          ) {
            const transitionType =
              currentSegment.transition_out || "cross_dissolve";
            const transitionProgress = 1 - timeUntilEnd / transitionDuration;

            console.log(
              `[DEBUG] Applying transition: ${transitionType} at ${(
                transitionProgress * 100
              ).toFixed(1)}%`
            );

            // Apply transition effect
            switch (transitionType) {
              case "cross_dissolve":
                applyCrossDissolveTransition(ctx, transitionProgress);
                break;
              case "fade_in":
                // Fade in from black (reverse of fade out)
                const fadeInIntensity = transitionProgress;
                applyBrightnessEffect(
                  ctx,
                  canvas.width,
                  canvas.height,
                  fadeInIntensity * transitionIntensity
                );
                break;
              case "fade_out":
                applyCrossDissolveTransition(ctx, transitionProgress);
                break;
              case "slide":
                applySlideTransition(ctx, transitionProgress);
                break;
              case "zoom":
                applyZoomTransition(ctx, transitionProgress);
                break;
              case "whip_pan":
                applyWhipPanTransition(ctx, transitionProgress);
                break;
              case "spin":
                applySpinTransition(ctx, transitionProgress);
                break;
              case "glitch":
                applyGlitchTransition(ctx, transitionProgress);
                break;
            }
          }
        }
      } else {
        setActiveEffects([]);
      }

      // Draw debug info on top of everything
      const rectX = 10 + ((video.currentTime * 10) % (canvas.width - 60));
      ctx.fillStyle = "red";
      ctx.fillRect(rectX, 10, 50, 50);

      // Add time display to canvas
      ctx.fillStyle = "white";
      ctx.font = "20px Arial";
      ctx.fillText(`Time: ${video.currentTime.toFixed(2)}s`, 10, 50);
      ctx.fillText(`RectX: ${rectX.toFixed(0)}`, 10, 80);
      ctx.fillText(`Playing: ${!video.paused}`, 10, 110);
      ctx.fillText(`Ready: ${video.readyState}`, 10, 140);
      ctx.fillText(
        `Active: ${currentSegment ? currentSegment.tags.join(",") : "none"}`,
        10,
        170
      );
    },
    [segments, setActiveEffects]
  );

  // --- Manual Animation Loop that doesn't rely on video events ---
  useEffect(() => {
    const originalVideo = originalVideoRef.current;
    if (!originalVideo) return;

    let animationId: number | null = null;
    let isAnimating = false;
    let lastSpeedRate = 1.0;
    let lastSegmentIndex = -1;
    let lastVideoTime = 0;

    // Simple speed effect handling - no complex transitions
    const currentSegment = segments.find(
      (segment) => currentTime >= segment.start && currentTime < segment.end
    );

    if (currentSegment) {
      const enabledEffects = currentSegment.tags.filter((tag) => {
        const effectCustomization = currentSegment.effectCustomizations?.[tag];
        return effectCustomization?.enabled === true;
      });

      const speedEffect = enabledEffects.find(
        (effect) =>
          effect === "speed_up" ||
          effect === "slow_motion" ||
          effect === "speed"
      );

      if (speedEffect) {
        const effectCustomization =
          currentSegment.effectCustomizations?.[speedEffect];
        let targetSpeed = 1.0;

        if (speedEffect === "speed_up") {
          targetSpeed = effectCustomization?.parameters?.speed_factor || 2.0;
        } else if (speedEffect === "slow_motion") {
          targetSpeed = effectCustomization?.parameters?.speed_factor || 0.5;
        } else if (speedEffect === "speed") {
          targetSpeed = effectCustomization?.parameters?.speed_factor || 1.0;
        }

        // Apply speed directly without transitions
        if (Math.abs(originalVideo.playbackRate - targetSpeed) > 0.01) {
          originalVideo.playbackRate = targetSpeed;
          console.log(
            `[DEBUG] Applied ${speedEffect} with speed ${targetSpeed}`
          );
        }
      } else {
        // Reset to normal speed if no speed effect
        if (Math.abs(originalVideo.playbackRate - 1.0) > 0.01) {
          originalVideo.playbackRate = 1.0;
          console.log("[DEBUG] Reset to normal speed");
        }
      }
    } else {
      // Reset to normal speed if not in any segment
      if (Math.abs(originalVideo.playbackRate - 1.0) > 0.01) {
        originalVideo.playbackRate = 1.0;
        console.log("[DEBUG] Reset to normal speed (no segment)");
      }
    }

    function manualAnimationLoop() {
      if (!originalVideo) {
        isAnimating = false;
        return;
      }

      // Update state with current video time
      const currentTime = originalVideo.currentTime;
      setCurrentTime(currentTime);

      // Use currentTime directly for segment detection (no normalized time)
      const currentSegmentIndex = segments.findIndex(
        (segment) => currentTime >= segment.start && currentTime < segment.end
      );
      const currentSegment =
        currentSegmentIndex >= 0 ? segments[currentSegmentIndex] : null;
      const nextSegment = segments.find(
        (segment) =>
          segment.start > currentTime && segment.start <= currentTime + 0.5
      );

      // Handle speed effects when segment changes
      if (currentSegmentIndex !== lastSegmentIndex) {
        lastSegmentIndex = currentSegmentIndex;

        if (currentSegment) {
          const enabledEffects = currentSegment.tags.filter((tag) => {
            const effectCustomization =
              currentSegment.effectCustomizations?.[tag];
            return effectCustomization?.enabled === true;
          });

          const speedEffect = enabledEffects.find(
            (effect) =>
              effect === "speed_up" ||
              effect === "slow_motion" ||
              effect === "speed"
          );

          if (speedEffect) {
            const effectCustomization =
              currentSegment.effectCustomizations?.[speedEffect];
            let targetSpeed = 1.0;

            if (speedEffect === "speed_up") {
              targetSpeed =
                effectCustomization?.parameters?.speed_factor || 2.0;
            } else if (speedEffect === "slow_motion") {
              targetSpeed =
                effectCustomization?.parameters?.speed_factor || 0.5;
            } else if (speedEffect === "speed") {
              targetSpeed =
                effectCustomization?.parameters?.speed_factor || 1.0;
            }

            originalVideo.playbackRate = targetSpeed;
            console.log(
              `[DEBUG] Segment ${currentSegmentIndex}: Applied ${speedEffect} with speed ${targetSpeed}`
            );
          } else {
            // Reset to normal speed if no speed effect in this segment
            originalVideo.playbackRate = 1.0;
            console.log(
              `[DEBUG] Segment ${currentSegmentIndex}: Reset to normal speed`
            );
          }
        } else {
          // Reset to normal speed if not in any segment
          originalVideo.playbackRate = 1.0;
          console.log("[DEBUG] No segment: Reset to normal speed");
        }
      }
      const isNearSegmentBoundary =
        nextSegment && nextSegment.start - currentTime <= 0.5;
      const isInTransitionPeriod =
        isNearSegmentBoundary && nextSegment && currentSegment;

      // Update segment tracking for other logic
      if (currentSegmentIndex !== lastSegmentIndex) {
        lastSegmentIndex = currentSegmentIndex;
      }

      // Apply effects
      applyRealTimeEffects(currentTime);

      // Continue animation if still playing and not ended
      if (isAnimating && !originalVideo.paused && !originalVideo.ended) {
        animationId = requestAnimationFrame(manualAnimationLoop);
      } else {
        isAnimating = false;
      }
    }

    // Start animation when video is playing
    if (
      isPlaying &&
      !originalVideo.paused &&
      !originalVideo.ended &&
      !isAnimating
    ) {
      console.log("[DEBUG] Starting manual animation loop");
      console.log("[DEBUG] Video state:", {
        paused: originalVideo.paused,
        ended: originalVideo.ended,
        currentTime: originalVideo.currentTime,
        readyState: originalVideo.readyState,
        isPlaying: isPlaying,
      });
      isAnimating = true;
      manualAnimationLoop();
    }

    // Stop animation when video is paused or ended
    if (
      (!isPlaying || originalVideo.paused || originalVideo.ended) &&
      isAnimating
    ) {
      console.log("[DEBUG] Stopping manual animation loop");
      isAnimating = false;
      if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
      isAnimating = false;
    };
  }, [isPlaying, applyRealTimeEffects, segments]);

  const applyBrightnessEffect = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    factor: number
  ) => {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, data[i] * factor);
      data[i + 1] = Math.min(255, data[i + 1] * factor);
      data[i + 2] = Math.min(255, data[i + 2] * factor);
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const applyContrastEffect = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    factor: number
  ) => {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, Math.max(0, (data[i] - 128) * factor + 128));
      data[i + 1] = Math.min(
        255,
        Math.max(0, (data[i + 1] - 128) * factor + 128)
      );
      data[i + 2] = Math.min(
        255,
        Math.max(0, (data[i + 2] - 128) * factor + 128)
      );
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const applySaturationEffect = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    factor: number
  ) => {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      data[i] = Math.min(255, gray + (data[i] - gray) * factor);
      data[i + 1] = Math.min(255, gray + (data[i + 1] - gray) * factor);
      data[i + 2] = Math.min(255, gray + (data[i + 2] - gray) * factor);
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const applyBlurEffect = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    radius: number
  ) => {
    ctx.filter = `blur(${radius}px)`;
    ctx.drawImage(ctx.canvas, 0, 0);
    ctx.filter = "none";
  };

  const applyVintageEffect = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number
  ) => {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, data[i] * 1.1);
      data[i + 1] = Math.min(255, data[i + 1] * 0.9);
      data[i + 2] = Math.min(255, data[i + 2] * 0.8);
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const applyBlackWhiteEffect = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number
  ) => {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      data[i] = gray;
      data[i + 1] = gray;
      data[i + 2] = gray;
    }
    ctx.putImageData(imageData, 0, 0);
  };

  // Transition effect helper functions
  const applyCrossDissolveTransition = (
    ctx: CanvasRenderingContext2D,
    progress: number
  ) => {
    const fadeIntensity = 1 - progress;
    applyBrightnessEffect(
      ctx,
      ctx.canvas.width,
      ctx.canvas.height,
      fadeIntensity
    );
  };

  const applySlideTransition = (
    ctx: CanvasRenderingContext2D,
    progress: number
  ) => {
    const slideOffset = progress * ctx.canvas.width;
    const imageData = ctx.getImageData(
      0,
      0,
      ctx.canvas.width,
      ctx.canvas.height
    );

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = ctx.canvas.width;
    tempCanvas.height = ctx.canvas.height;
    const tempCtx = tempCanvas.getContext("2d");

    if (tempCtx) {
      tempCtx.putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.drawImage(tempCanvas, -slideOffset, 0);
    }
  };

  const applyZoomTransition = (
    ctx: CanvasRenderingContext2D,
    progress: number
  ) => {
    const scale = 1 + progress * 0.5;
    const centerX = ctx.canvas.width / 2;
    const centerY = ctx.canvas.height / 2;

    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.scale(scale, scale);
    ctx.translate(-centerX, -centerY);
    ctx.drawImage(ctx.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();
  };

  const applyWhipPanTransition = (
    ctx: CanvasRenderingContext2D,
    progress: number
  ) => {
    const panOffset = progress * ctx.canvas.width * 2;
    const imageData = ctx.getImageData(
      0,
      0,
      ctx.canvas.width,
      ctx.canvas.height
    );

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = ctx.canvas.width;
    tempCanvas.height = ctx.canvas.height;
    const tempCtx = tempCanvas.getContext("2d");

    if (tempCtx) {
      tempCtx.putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.drawImage(tempCanvas, panOffset, 0);
    }
  };

  const applySpinTransition = (
    ctx: CanvasRenderingContext2D,
    progress: number
  ) => {
    const rotation = progress * Math.PI * 2;
    const centerX = ctx.canvas.width / 2;
    const centerY = ctx.canvas.height / 2;

    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(rotation);
    ctx.translate(-centerX, -centerY);
    ctx.drawImage(ctx.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();
  };

  const applyGlitchTransition = (
    ctx: CanvasRenderingContext2D,
    progress: number
  ) => {
    const imageData = ctx.getImageData(
      0,
      0,
      ctx.canvas.width,
      ctx.canvas.height
    );
    const data = imageData.data;

    // Random glitch lines based on progress
    const glitchIntensity = Math.floor(progress * 20);
    for (let i = 0; i < glitchIntensity; i++) {
      const y = Math.floor(Math.random() * ctx.canvas.height);
      const offset = Math.floor(Math.random() * 20) - 10;

      for (let x = 0; x < ctx.canvas.width; x++) {
        const srcIndex = (y * ctx.canvas.width + x) * 4;
        const dstIndex =
          (y * ctx.canvas.width +
            Math.max(0, Math.min(ctx.canvas.width - 1, x + offset))) *
          4;

        data[dstIndex] = data[srcIndex];
        data[dstIndex + 1] = data[srcIndex + 1];
        data[dstIndex + 2] = data[srcIndex + 2];
        data[dstIndex + 3] = data[srcIndex + 3];
      }
    }
    ctx.putImageData(imageData, 0, 0);
  };

  // Fix: Always enable play button if video is loaded
  const isVideoLoaded = duration > 0;

  const togglePlay = async () => {
    // Prevent rapid clicking
    if (isProcessingPlayToggle) {
      console.log("[DEBUG] Play toggle already processing, ignoring click");
      return;
    }

    console.log("[DEBUG] togglePlay called", { isPlaying, duration });

    setIsProcessingPlayToggle(true);

    if (originalVideoRef.current) {
      const video = originalVideoRef.current;

      // Check if video is ready
      if (video.readyState < 2) {
        console.log("[DEBUG] Video not ready, readyState:", video.readyState);
        setIsProcessingPlayToggle(false);
        return;
      }

      if (isPlaying) {
        console.log("[DEBUG] Pausing video");
        video.pause();
        // State will be updated by the pause event listener
      } else {
        console.log("[DEBUG] Playing video");
        try {
          // Check if video has a valid source
          if (!video.src && !video.currentSrc) {
            console.error("[DEBUG] Video has no source");
            setIsProcessingPlayToggle(false);
            return;
          }

          // Force the video to play
          await video.play();
          console.log("[DEBUG] Video play() completed successfully");

          // State will be updated by the play event listener
          // The animation loop will automatically start when isPlaying becomes true
        } catch (error) {
          console.error("[DEBUG] Video play() failed:", error);
          setIsProcessingPlayToggle(false);
        }
      }
    } else {
      console.log("[DEBUG] No video ref available");
      setIsProcessingPlayToggle(false);
    }

    // Allow next click after a short delay
    setTimeout(() => {
      setIsProcessingPlayToggle(false);
    }, 300);
  };

  const handleVolumeChange = (newVolume: number) => {
    setVolume(newVolume);
    if (originalVideoRef.current) {
      originalVideoRef.current.volume = newVolume;
    }
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (originalVideoRef.current) {
      originalVideoRef.current.muted = !isMuted;
    }
  };

  // Sync seeking
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;
    if (originalVideoRef.current) {
      // Set video to the new time
      originalVideoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
      onPlaybackTimeChange(newTime);

      // Sync real time with video time when seeking

      // Effects will be applied automatically by the animation loop
    }
  };

  const applySuggestion = (suggestion: LLMSuggestion) => {
    console.log("[DEBUG] Applying suggestion:", suggestion);

    if (suggestion.segment_data) {
      const llmSegment: EditDecisionSegment = {
        start: suggestion.segment_data.start,
        end: suggestion.segment_data.end,
        tags: [suggestion.segment_data.effect],
        transition: undefined,
        transition_duration: undefined,
        speed: suggestion.segment_data.intensity,
        transition_in: undefined,
        transition_out: undefined,
      };

      console.log("[DEBUG] Creating segment:", llmSegment);
      onSegmentUpdate(segments.length, llmSegment);

      // Mark suggestion as applied
      suggestion.applied = true;

      // Jump to the start of the effect and start playing
      if (originalVideoRef.current) {
        originalVideoRef.current.currentTime = suggestion.segment_data.start;
        setCurrentTime(suggestion.segment_data.start);

        // Start playing if not already playing
        if (!isPlaying) {
          originalVideoRef.current
            .play()
            .then(() => {
              console.log("[DEBUG] Started playing after applying suggestion");
              // Animation loop will handle effects automatically
            })
            .catch((err) => {
              console.error("[DEBUG] Failed to start playing:", err);
            });
        } else {
          // If already playing, animation loop will handle effects automatically
        }

        // Animation loop will handle effects automatically
      }
    }
  };

  // Initialize canvas with black background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      console.log("[DEBUG] Initializing canvas");
      const ctx = canvas.getContext("2d");
      if (ctx) {
        // Set a default size
        canvas.width = 640;
        canvas.height = 480;

        // Draw black background
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        console.log("[DEBUG] Canvas initialized with black background");
      }
    }
  }, []);

  // Initialize canvas when video loads and add video event listeners
  useEffect(() => {
    const video = originalVideoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      const handleCanPlay = () => {
        console.log(
          "🎬 VideoTimelineEditor: Video can play, initializing canvas"
        );
        console.log("🎬 VideoTimelineEditor: Video duration:", video.duration);
        console.log(
          "🎬 VideoTimelineEditor: Video dimensions:",
          video.videoWidth,
          "x",
          video.videoHeight
        );
        console.log(
          "🎬 VideoTimelineEditor: Video readyState:",
          video.readyState
        );
        console.log("🎬 VideoTimelineEditor: Video src:", video.src);
        console.log(
          "🎬 VideoTimelineEditor: Video currentSrc:",
          video.currentSrc
        );

        if (video.videoWidth > 0 && video.videoHeight > 0) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          console.log(
            "🎬 VideoTimelineEditor: Canvas initialized with size:",
            canvas.width,
            "x",
            canvas.height
          );

          // Initialize real time tracking
        } else {
          console.log("🎬 VideoTimelineEditor: Video dimensions not ready yet");
        }
      };

      const handlePlay = () => {
        console.log("[DEBUG] Video play event fired");
        setIsPlaying(true);
        // Start real time tracking
      };

      const handlePause = () => {
        console.log("[DEBUG] Video pause event fired");
        setIsPlaying(false);
        // Stop real time tracking and save current position
      };

      const handleEnded = () => {
        console.log("[DEBUG] Video ended event fired");
        setIsPlaying(false);
      };

      const handleTimeUpdate = () => {
        // Keep current time in sync
        setCurrentTime(video.currentTime);
        onPlaybackTimeChange(video.currentTime);

        // Update real time for smooth playhead movement
        if (isPlaying) {
          const now = Date.now();
        }
      };

      const handleError = (error: Event) => {
        console.error("🎬 VideoTimelineEditor: Video error:", error);
        console.error(
          "🎬 VideoTimelineEditor: Video error details:",
          video.error
        );
      };

      const handleLoadStart = () => {
        console.log("🎬 VideoTimelineEditor: Video load started");
      };

      const handleLoadedData = () => {
        console.log("🎬 VideoTimelineEditor: Video data loaded");
      };

      video.addEventListener("canplay", handleCanPlay);
      video.addEventListener("play", handlePlay);
      video.addEventListener("pause", handlePause);
      video.addEventListener("ended", handleEnded);
      video.addEventListener("timeupdate", handleTimeUpdate);
      video.addEventListener("error", handleError);
      video.addEventListener("loadstart", handleLoadStart);
      video.addEventListener("loadeddata", handleLoadedData);

      return () => {
        video.removeEventListener("canplay", handleCanPlay);
        video.removeEventListener("play", handlePlay);
        video.removeEventListener("pause", handlePause);
        video.removeEventListener("ended", handleEnded);
        video.removeEventListener("timeupdate", handleTimeUpdate);
        video.removeEventListener("error", handleError);
        video.removeEventListener("loadstart", handleLoadStart);
        video.removeEventListener("loadeddata", handleLoadedData);
      };
    }
  }, [applyRealTimeEffects]);

  // Initialize segments with enabled and effectCustomizations properties
  useEffect(() => {
    if (segments.length > 0) {
      const initializedSegments = segments.map((segment, index) => {
        // If segment doesn't have enabled property, initialize it
        if (segment.enabled === undefined) {
          segment.enabled = true;
        }

        // If segment doesn't have effectCustomizations, initialize it
        if (!segment.effectCustomizations) {
          const customizations: {
            [effectName: string]: {
              enabled: boolean;
              parameters: { [paramName: string]: any };
            };
          } = {};

          // Initialize all effects as enabled with default parameters
          segment.tags.forEach((effectName) => {
            customizations[effectName] = {
              enabled: true,
              parameters: {},
            };
          });

          segment.effectCustomizations = customizations;
        }

        return segment;
      });

      // Update segments if they were initialized
      if (JSON.stringify(initializedSegments) !== JSON.stringify(segments)) {
        console.log(
          "Initializing segments with enabled and effectCustomizations:",
          initializedSegments
        );
        initializedSegments.forEach((segment, index) => {
          onSegmentUpdate(index, segment);
        });
      }
    }
  }, [segments]);

  // Add this useEffect to recalculate current segment and active effects when segments change
  useEffect(() => {
    // Recalculate current segment index
    const segmentIndex = getSegmentAtTime(currentTime);
    setCurrentSegmentIndex(segmentIndex);

    // Update active effects for the new segment
    const currentSegment = segments.find(
      (segment) => currentTime >= segment.start && currentTime <= segment.end
    );
    if (currentSegment && currentSegment.tags.length > 0) {
      setActiveEffects(currentSegment.tags);
    } else {
      setActiveEffects([]);
    }
  }, [segments, currentTime]);

  // New: Segment editing functions
  const handleEditSegment = (segment: EditDecisionSegment, index: number) => {
    setEditingSegment(segment);
    setEditingSegmentIndex(index);
  };

  const handleSegmentChange = (updatedSegment: EditDecisionSegment) => {
    if (editingSegmentIndex !== null) {
      console.log("Updating segment:", editingSegmentIndex, updatedSegment);

      // Update the segments array
      const updatedSegments = [...segments];
      updatedSegments[editingSegmentIndex] = updatedSegment;

      // Call the parent's onSegmentUpdate
      onSegmentUpdate(editingSegmentIndex, updatedSegment);

      // Close the editor
      setEditingSegment(null);
      setEditingSegmentIndex(-1);
    }
  };

  const handleCloseSegmentEditor = () => {
    setEditingSegment(null);
    setEditingSegmentIndex(-1);
  };

  // Helper function to get current segment index
  const getCurrentSegmentIndex = (time: number) => {
    return segments.findIndex(
      (segment) => time >= segment.start && time < segment.end
    );
  };

  // LLM timeline testing functions
  const testLLMTimeline = async () => {
    if (!projectId) {
      console.error("Project ID is required for LLM testing");
      return;
    }

    setIsTestingLLM(true);
    try {
      const response = await fetch(
        `/api/v1/multi-video/projects/${projectId}/test-llm-timeline`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            edit_scale: 0.7,
            style_preferences: {},
            cross_video_effects: {},
            target_duration: null,
          }),
        }
      );

      if (response.ok) {
        const result = await response.json();
        setLlmTestResult(result);
        setShowLLMResults(true);
        console.log("LLM timeline test completed:", result);
      } else {
        console.error("Failed to test LLM timeline");
      }
    } catch (error) {
      console.error("Error testing LLM timeline:", error);
    } finally {
      setIsTestingLLM(false);
    }
  };

  const getLLMTestResult = async () => {
    if (!projectId) return;

    try {
      const response = await fetch(
        `/api/v1/multi-video/projects/${projectId}/llm-test-result`
      );
      if (response.ok) {
        const result = await response.json();
        setLlmTestResult(result);
        setShowLLMResults(true);
      }
    } catch (error) {
      console.error("Error getting LLM test result:", error);
    }
  };

  return (
    <>
      <div className="w-full h-full">
        {/* Main Content Area - Two Column Layout */}
        <div className="w-full">
          {/* Main Content Area - Full Width */}
          <div className="space-y-6">
            {/* Video Previews Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Original Video */}
              <div className="space-y-3">
                <div className="relative bg-black rounded-xl overflow-hidden shadow-lg border-2 border-blue-500">
                  {convertedVideoUrl ? (
                    <video
                      ref={originalVideoRef}
                      className={`w-full object-contain ${
                        videoOrientation === "vertical"
                          ? "h-[1000px]"
                          : "h-[900px]"
                      }`}
                      src={convertedVideoUrl}
                      crossOrigin="anonymous"
                      onLoadedMetadata={() => {
                        const video = originalVideoRef.current;
                        if (video) {
                          console.log(
                            "🔧 [VideoTimelineEditor] onLoadedMetadata - customDuration:",
                            customDuration,
                            "video.duration:",
                            video.duration
                          );
                          // For multi-video projects, use the actual video duration instead of customDuration
                          // because the rendered video might have a different duration due to effects
                          console.log(
                            "🔧 [VideoTimelineEditor] Using actual video.duration:",
                            video.duration
                          );
                          setDuration(video.duration);
                          setVideoOrientation(
                            video.videoWidth > video.videoHeight
                              ? "horizontal"
                              : "vertical"
                          );
                        }
                      }}
                      onCanPlay={() => {
                        const video = originalVideoRef.current;
                        const canvas = canvasRef.current;
                        if (
                          video &&
                          canvas &&
                          video.videoWidth > 0 &&
                          video.videoHeight > 0
                        ) {
                          canvas.width = video.videoWidth;
                          canvas.height = video.videoHeight;
                          console.log(
                            "[DEBUG] Canvas initialized with size:",
                            canvas.width,
                            "x",
                            canvas.height
                          );
                          // Draw initial frame
                          applyRealTimeEffects();
                        }
                      }}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-[704px] text-gray-400">
                      <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-4 border-2 border-gray-300 border-dashed rounded-lg flex items-center justify-center">
                          <Video className="w-8 h-8" />
                        </div>
                        <p>Video loading...</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Effects Video */}
              <div className="space-y-3">
                <div className="relative bg-black rounded-xl overflow-hidden shadow-lg border-2 border-green-500">
                  <canvas
                    ref={canvasRef}
                    className={`w-full object-contain ${
                      videoOrientation === "vertical"
                        ? "h-[1000px]"
                        : "h-[900px]"
                    }`}
                  />
                </div>
              </div>
            </div>

            {/* Integrated Playback Interface */}
            <div className="bg-white rounded-xl p-6 shadow-lg border">
              {/* Header with integrated controls */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-6">
                  <h3 className="text-xl font-semibold text-gray-900">
                    Video Timeline
                  </h3>
                  <button
                    onClick={togglePlay}
                    disabled={isProcessingPlayToggle}
                    className="flex items-center justify-center w-10 h-10 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-full transition-colors"
                  >
                    {isPlaying ? (
                      <Pause className="w-4 h-4" />
                    ) : (
                      <Play className="w-4 h-4 ml-0.5" />
                    )}
                  </button>
                  <div className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                    {segments.length} segments
                  </div>

                  {/* LLM Timeline Testing */}
                  {projectId && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={testLLMTimeline}
                        disabled={isTestingLLM}
                        className="flex items-center gap-2 px-3 py-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white text-sm rounded-full transition-colors"
                      >
                        <Sparkles className="w-3 h-3" />
                        {isTestingLLM ? "Testing..." : "Test LLM"}
                      </button>

                      <button
                        onClick={getLLMTestResult}
                        className="flex items-center gap-2 px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-full transition-colors"
                      >
                        <Eye className="w-3 h-3" />
                        View Results
                      </button>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <Volume2 className="w-4 h-4 text-gray-600" />
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={volume}
                      onChange={(e) =>
                        handleVolumeChange(parseFloat(e.target.value))
                      }
                      className="w-16 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <button
                      onClick={toggleMute}
                      className="p-1 hover:bg-gray-100 rounded transition-colors"
                    >
                      {isMuted ? (
                        <VolumeX className="w-3 h-3 text-gray-600" />
                      ) : (
                        <Volume2 className="w-3 h-3 text-gray-600" />
                      )}
                    </button>
                  </div>
                  <div className="text-sm text-gray-500">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </div>
                </div>
              </div>

              {/* Timeline */}
              <div
                className="relative bg-gray-100 rounded-lg p-4 h-28 cursor-pointer border-2 border-gray-200 overflow-hidden"
                onClick={handleTimelineClick}
              >
                {/* Transition indicators positioned at the top of timeline */}
                {segments.map((segment, index) => {
                  if (index === 0) return null; // No transition before first segment

                  const prevSegment = segments[index - 1];
                  const transitionOut =
                    prevSegment.transition_out || prevSegment.transition;
                  const transitionIn =
                    segment.transition_in || segment.transition;

                  // Use the transition from the previous segment's transition_out or current segment's transition_in
                  const transitionType = transitionOut || transitionIn;

                  if (!transitionType || transitionType === "none") return null;

                  // Calculate position based on the segment boundary (start of current segment)
                  const transitionPosition = (segment.start / duration) * 100;
                  const transitionWidth = 50; // Fixed width for transition indicator

                  // Get transition duration from the previous segment
                  const transitionDuration =
                    prevSegment.transition_duration || 0.5;

                  // Calculate bar width based on timeline scale (timeline width / total duration * transition duration)
                  const timelineWidth = 100; // Timeline width in percentage
                  const durationWidth =
                    (timelineWidth / duration) * transitionDuration;

                  return (
                    <div
                      key={`transition-${index}`}
                      className="absolute top-0 transform -translate-x-1/2 cursor-pointer group z-20"
                      style={{
                        left: `${transitionPosition}%`,
                        width: `${transitionWidth}px`,
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        // Handle transition click - open transition editor
                        setEditingTransitionIndex(index);
                        setShowTransitionEditor(true);
                      }}
                    >
                      {/* TikTok-style transition effect */}
                      <div className="relative bg-blue-600 rounded-lg px-2 py-1 shadow-md border border-blue-700 transform hover:scale-110 transition-all duration-200 group-hover:shadow-lg">
                        {/* Left arrow */}
                        <div className="absolute left-1 top-1/2 transform -translate-y-1/2 w-0 h-0 border-t-2 border-b-2 border-l-4 border-transparent border-l-white"></div>

                        {/* Center bar */}
                        <div
                          className="h-1 bg-white rounded-full mx-2"
                          style={{ width: `${durationWidth}px` }}
                        ></div>

                        {/* Right arrow */}
                        <div className="absolute right-1 top-1/2 transform -translate-y-1/2 w-0 h-0 border-t-2 border-b-2 border-r-4 border-transparent border-r-white"></div>

                        {/* Transition type label */}
                        <div className="text-xs text-white font-medium text-center mt-1">
                          {transitionType}
                        </div>
                      </div>
                    </div>
                  );
                })}

                {/* Progress overlay */}
                <div
                  className="absolute top-0 left-0 h-full bg-blue-500/30 rounded-lg"
                  style={{
                    width: `${(currentTime / duration) * 100}%`,
                  }}
                />

                {/* Segments */}
                {segments.map((segment, index) => {
                  // Debug logging for segment visibility
                  const segmentWidth =
                    ((segment.end - segment.start) / duration) * 100;
                  const segmentLeft = (segment.start / duration) * 100;

                  console.log(
                    `🔧 [VideoTimelineEditor] Segment ${index + 1}:`,
                    {
                      start: segment.start,
                      end: segment.end,
                      duration: segment.end - segment.start,
                      left: `${segmentLeft}%`,
                      width: `${segmentWidth}%`,
                      visible: segmentWidth > 0.5, // Consider segments with width > 0.5% as visible
                      effects: segment.tags?.length || 0,
                    }
                  );

                  return (
                    <div
                      key={index}
                      className={`absolute top-0 h-full border-l-2 ${
                        index === currentSegmentIndex
                          ? "border-blue-600 bg-blue-100/50"
                          : "border-gray-400 bg-gray-300/50"
                      }`}
                      style={{
                        left: `${segmentLeft}%`,
                        width: `${segmentWidth}%`,
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        setCurrentSegmentIndex(index);
                      }}
                    >
                      {/* Segment content */}
                      <div className="relative h-full flex items-center justify-between px-2">
                        <div className="text-xs text-gray-700 truncate font-medium">
                          {formatTime(segment.start)} -{" "}
                          {formatTime(segment.end)}
                        </div>
                        <button
                          className="p-2.5 hover:bg-white/50 rounded transition-all duration-200 bg-white/60 shadow-sm hover:shadow-md hover:scale-105"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleEditSegment(segment, index);
                          }}
                          title="Edit segment effects"
                        >
                          <Settings className="w-4 h-4 text-gray-700" />
                        </button>
                      </div>

                      {/* Effect indicators */}
                      {segment.tags && segment.tags.length > 0 && (
                        <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-xs px-2 py-1 truncate text-center">
                          {segment.tags.length} effects
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* Playhead */}
                <div
                  className="absolute top-0 w-1 h-full bg-red-500 rounded-full shadow-lg z-10"
                  style={{
                    left: `${(currentTime / duration) * 100}%`,
                  }}
                />
              </div>

              {/* Controls row */}
              {/* Active Effects Display */}
              {activeEffects.length > 0 && (
                <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-600" />
                    <span className="font-medium text-green-800">
                      Active Effects:
                    </span>
                    <span className="text-green-700">
                      {activeEffects.join(", ")}
                    </span>
                  </div>
                </div>
              )}

              {/* AI Suggestions integrated within the timeline box */}
              <div className="mt-4 border-t pt-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-3">
                  AI Suggestions by Segment
                </h4>
                <div
                  className={`grid gap-3 ${
                    segments.length <= 3
                      ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
                      : segments.length <= 6
                      ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
                      : segments.length <= 8
                      ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5"
                      : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6"
                  }`}
                >
                  {segments.map((segment, segmentIndex) => {
                    // Get suggestions for this specific segment
                    const segmentSuggestions = llmSuggestions.filter(
                      (suggestion) => suggestion.segment_index === segmentIndex
                    );

                    return (
                      <div
                        key={segmentIndex}
                        className="border border-gray-200 rounded-lg p-2 min-w-0 bg-gray-50"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <h5 className="font-medium text-gray-900 text-xs truncate">
                            Segment {segmentIndex + 1}:{" "}
                            {formatTime(segment.start)} -{" "}
                            {formatTime(segment.end)}
                          </h5>
                          <span className="text-xs text-gray-500 flex-shrink-0 ml-2">
                            {segmentSuggestions.length} suggestions
                          </span>
                        </div>

                        {segmentSuggestions.length > 0 ? (
                          <div className="space-y-1">
                            {segmentSuggestions.map(
                              (suggestion, suggestionIndex) => (
                                <div
                                  key={suggestionIndex}
                                  className="bg-white rounded p-1.5 border"
                                >
                                  <div className="flex items-start justify-between mb-1">
                                    <h6 className="font-semibold text-xs text-gray-900 truncate flex-1 mr-2">
                                      {suggestion.title}
                                    </h6>
                                    <span className="text-xs bg-blue-100 text-blue-800 px-1 py-0.5 rounded-full flex-shrink-0">
                                      {Math.round(suggestion.confidence * 100)}%
                                    </span>
                                  </div>
                                  <p className="text-xs text-gray-600 mb-1 line-clamp-2">
                                    {suggestion.description}
                                  </p>
                                </div>
                              )
                            )}
                            {/* Single Customize button per segment */}
                            <Button
                              size="sm"
                              variant="outline"
                              className="w-full text-xs"
                              onClick={() =>
                                handleEditSegment(segment, segmentIndex)
                              }
                            >
                              Customize Segment
                            </Button>
                          </div>
                        ) : (
                          <div className="text-xs text-gray-500 text-center py-2">
                            No suggestions available
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Action Buttons - Moved inside the white content area */}
              {onBack && onProceed && (
                <div className="flex justify-between pt-6 border-t mt-6">
                  <Button
                    onClick={onBack}
                    className="px-6 bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300 hover:border-gray-400"
                  >
                    ← Back
                  </Button>
                  <Button
                    onClick={async (event) => {
                      // Show loading state
                      const button = event?.target as HTMLButtonElement;
                      const originalText = button.innerHTML;
                      button.innerHTML = "🎬 Rendering...";
                      button.disabled = true;

                      try {
                        console.log("🎬 Starting custom effects rendering...");
                        console.log("📊 Segments to render:", segments);

                        // Import the API client
                        const { apiClient } = await import("@/lib/api");

                        let result;

                        // Determine if this is a multi-video project or single video
                        if (projectId) {
                          // Multi-video project
                          console.log(
                            "🎬 Rendering multi-video project:",
                            projectId
                          );
                          result =
                            await apiClient.downloadMultiVideoWithCustomEffects(
                              projectId,
                              segments,
                              "high" // quality preset
                            );
                        } else {
                          // Check if this is a multi-video project
                          if (projectId) {
                            // Multi-video project - use project ID
                            console.log(
                              "🎬 Rendering multi-video project:",
                              projectId
                            );
                            result =
                              await apiClient.downloadMultiVideoWithCustomEffects(
                                projectId,
                                segments,
                                "high"
                              );
                          } else {
                            // Single video - extract video ID from convertedVideoUrl
                            const videoIdMatch =
                              convertedVideoUrl?.match(/\/videos\/([^\/]+)\//);
                            if (videoIdMatch) {
                              const videoId = videoIdMatch[1];
                              console.log(
                                "🎬 Rendering single video:",
                                videoId
                              );
                              result =
                                await apiClient.downloadVideoWithCustomEffects(
                                  videoId,
                                  segments,
                                  "high" // quality preset
                                );
                            } else {
                              throw new Error(
                                "Could not extract video ID from URL"
                              );
                            }
                          }
                        }

                        if (result.success && result.download_url) {
                          console.log(
                            "✅ Video rendered successfully:",
                            result.download_url
                          );

                          // Show download link instead of auto-downloading (avoids CORS issues)
                          const downloadLink = document.createElement("div");
                          downloadLink.innerHTML = `
                            <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                                       background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); 
                                       z-index: 1000; text-align: center; max-width: 500px;">
                              <h3 style="margin: 0 0 15px 0; color: #2563eb;">🎉 Video Rendered Successfully!</h3>
                              <p style="margin: 0 0 20px 0; color: #374151;">Your video with custom effects has been rendered and is ready for download.</p>
                              <a href="${result.download_url}" 
                                 download="${
                                   result.filename || "custom-effects-video.mp4"
                                 }"
                                 style="display: inline-block; background: #2563eb; color: white; padding: 12px 24px; 
                                        text-decoration: none; border-radius: 6px; font-weight: 500; margin: 0 10px;">
                                📥 Download Video
                              </a>
                              <button onclick="this.parentElement.parentElement.remove()" 
                                      style="background: #6b7280; color: white; border: none; padding: 12px 24px; 
                                             border-radius: 6px; cursor: pointer; margin: 0 10px;">
                                ✕ Close
                              </button>
                              <p style="margin: 15px 0 0 0; font-size: 12px; color: #9ca3af;">
                                If the download doesn't start automatically, right-click the button and select "Save link as..."
                              </p>
                            </div>
                          `;
                          document.body.appendChild(downloadLink);

                          console.log(
                            "✅ Download link displayed successfully"
                          );
                        } else {
                          throw new Error(result.message || "Rendering failed");
                        }
                      } catch (error) {
                        console.error(
                          "❌ Error rendering video with custom effects:",
                          error
                        );

                        // Extract more specific error message
                        let errorMessage = "Unknown error occurred";
                        if (error instanceof Error) {
                          errorMessage = error.message;
                          // Try to extract more specific error from the message
                          if (
                            errorMessage.includes("500 Internal Server Error")
                          ) {
                            errorMessage =
                              "Video rendering failed. Please check that your video segments are within the video duration.";
                          } else if (errorMessage.includes("404 Not Found")) {
                            errorMessage =
                              "Video not found. Please try uploading the video again.";
                          } else if (errorMessage.includes("400 Bad Request")) {
                            errorMessage =
                              "Invalid segment data. Please check your video timeline.";
                          }
                        }

                        alert(`Failed to render video: ${errorMessage}`);
                      } finally {
                        // Restore button state
                        const button = event?.target as HTMLButtonElement;
                        if (button) {
                          button.innerHTML = originalText;
                          button.disabled = false;
                        }
                      }
                    }}
                    className="px-6 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
                  >
                    📥 Download Final Video
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Segment Editor Modal */}
      {editingSegment && editingSegmentIndex !== null && (
        <SegmentEditor
          segment={{
            id: editingSegmentIndex.toString(),
            startTime: editingSegment.start,
            endTime: editingSegment.end,
            duration: editingSegment.end - editingSegment.start,
            effects: editingSegment.tags || [],
            transitionIn: editingSegment.transition_in,
            transitionOut: editingSegment.transition_out,
            enabled: editingSegment.enabled,
            effectCustomizations: editingSegment.effectCustomizations,
          }}
          onSegmentChange={(updatedSegment) => {
            const updatedEditSegment: EditDecisionSegment = {
              ...editingSegment,
              start: updatedSegment.startTime,
              end: updatedSegment.endTime,
              tags: updatedSegment.effects || [],
              transition_in: updatedSegment.transitionIn,
              transition_out: updatedSegment.transitionOut,
              enabled: updatedSegment.enabled,
              effectCustomizations: updatedSegment.effectCustomizations,
            };
            handleSegmentChange(updatedEditSegment);
          }}
          onClose={handleCloseSegmentEditor}
        />
      )}

      {/* Transition Editor Modal */}
      {showTransitionEditor && editingTransitionIndex !== null && (
        <TransitionEditor
          segments={segments}
          transitionIndex={editingTransitionIndex}
          onSegmentUpdate={onSegmentUpdate}
          onClose={() => {
            setShowTransitionEditor(false);
            setEditingTransitionIndex(-1);
          }}
        />
      )}

      {/* LLM Test Results Modal */}
      {showLLMResults && llmTestResult && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-blue-900">
                LLM Timeline Test Results
              </h3>
              <button
                onClick={() => setShowLLMResults(false)}
                className="p-2 hover:bg-gray-100 rounded-full"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="text-sm font-semibold text-blue-900">
                  Segments Generated
                </h4>
                <p className="text-2xl font-bold text-blue-600">
                  {llmTestResult.details?.segments_count || 0}
                </p>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="text-sm font-semibold text-green-900">
                  Confidence Score
                </h4>
                <p className="text-2xl font-bold text-green-600">
                  {(
                    (llmTestResult.details?.confidence_score || 0) * 100
                  ).toFixed(1)}
                  %
                </p>
              </div>

              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="text-sm font-semibold text-purple-900">
                  Estimated Duration
                </h4>
                <p className="text-2xl font-bold text-purple-600">
                  {llmTestResult.details?.estimated_duration?.toFixed(1) || 0}s
                </p>
              </div>
            </div>

            {llmTestResult.details?.overall_strategy && (
              <div className="mb-4">
                <h4 className="font-semibold text-gray-900 mb-2">Strategy</h4>
                <p className="text-sm text-gray-700 bg-gray-100 p-3 rounded">
                  {llmTestResult.details.overall_strategy}
                </p>
              </div>
            )}

            {llmTestResult.details?.segment_assignments && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-2">
                  Segment Assignments
                </h4>
                <div className="space-y-2">
                  {llmTestResult.details.segment_assignments.map(
                    (assignment: any, index: number) => (
                      <div key={index} className="bg-gray-50 p-3 rounded">
                        <div className="flex justify-between items-center">
                          <span className="font-medium">
                            Segment {index + 1}
                          </span>
                          <span className="text-sm text-gray-500">
                            {assignment.source_video_id?.slice(0, 8)}...
                          </span>
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          {assignment.start_time?.toFixed(1)}s -{" "}
                          {assignment.end_time?.toFixed(1)}s
                        </div>
                        {assignment.llm_reasoning && (
                          <div className="text-xs text-gray-500 mt-1 italic">
                            "{assignment.llm_reasoning}"
                          </div>
                        )}
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default VideoTimelineEditor;
