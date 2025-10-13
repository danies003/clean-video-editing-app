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
  Target,
  Sparkles,
  Lightbulb,
  ArrowRight,
  Edit3,
  Layers,
  Clock,
  Music,
  Zap,
  Eye,
  EyeOff,
  RotateCcw,
  Save,
} from "lucide-react";

interface MultiVideoSegment {
  start: number;
  end: number;
  source_video_id: string; // UUID as string
  effects: string[];
  transition_in?: string;
  transition_out?: string;
  speed: number;
  volume: number;
  effectCustomizations?: { [key: string]: any };
  ai_recommendations?: {
    segment_reasoning: string;
    transition_reasoning: string;
    effects_reasoning: string;
    arrangement_reasoning: string;
    confidence_score: number;
    alternative_suggestions: string[];
  };
  // LLM timeline fields
  segment_order?: number;
  llm_reasoning?: string;
  confidence_score?: number;
  segment_tags?: string[];
}

interface MultiVideoPreviewProps {
  projectStatus: any; // Replace with actual MultiVideoProjectStatus type
  sourceVideos: Array<{
    id: string;
    url: string;
    name: string;
    duration: number;
  }>;
  timelineSegments: MultiVideoSegment[];
  onUpdateSegment: (
    segmentId: string,
    updates: Partial<MultiVideoSegment>
  ) => void;
  onReorderSegments: (newOrder: string[]) => void;
  onApplyChanges: () => void;
  onDiscardChanges: () => void;
  projectId?: string; // Add project ID for LLM testing
}

export const MultiVideoPreview: React.FC<MultiVideoPreviewProps> = ({
  projectStatus,
  sourceVideos,
  timelineSegments,
  onUpdateSegment,
  onReorderSegments,
  onApplyChanges,
  onDiscardChanges,
  projectId,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [selectedSegment, setSelectedSegment] = useState<string | null>(null);
  const [showAIRecommendations, setShowAIRecommendations] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // LLM timeline testing state
  const [isTestingLLM, setIsTestingLLM] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState<any>(null);
  const [showLLMResults, setShowLLMResults] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Cache clearing function
  const clearBrowserCache = useCallback(() => {
    try {
      // Clear localStorage
      localStorage.clear();

      // Clear sessionStorage
      sessionStorage.clear();

      // Clear any cached API responses
      if ("caches" in window) {
        caches.keys().then((names) => {
          names.forEach((name) => {
            caches.delete(name);
          });
        });
      }

      console.log("🧹 Browser cache cleared");
    } catch (error) {
      console.warn("Failed to clear browser cache:", error);
    }
  }, []);

  // Clear cache on component mount
  useEffect(() => {
    clearBrowserCache();
  }, [clearBrowserCache]);

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

  // Calculate total duration from segments
  useEffect(() => {
    const totalDuration = timelineSegments.reduce((total, segment) => {
      return total + (segment.end - segment.start);
    }, 0);
    setDuration(totalDuration);
  }, [timelineSegments]);

  // Video controls
  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    if (videoRef.current) {
      videoRef.current.volume = newVolume;
    }
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  // Timeline functions
  const getCurrentSegment = () => {
    let accumulatedTime = 0;
    for (const segment of timelineSegments) {
      const segmentDuration = segment.end - segment.start;
      if (
        currentTime >= accumulatedTime &&
        currentTime < accumulatedTime + segmentDuration
      ) {
        return segment;
      }
      accumulatedTime += segmentDuration;
    }
    return null;
  };

  const handleSegmentClick = (segmentId: string) => {
    setSelectedSegment(segmentId);
  };

  const handleSegmentEdit = (
    segmentId: string,
    updates: Partial<MultiVideoSegment>
  ) => {
    onUpdateSegment(segmentId, updates);
    setHasChanges(true);
  };

  const handleApplyChanges = () => {
    onApplyChanges();
    setHasChanges(false);
    setIsEditing(false);
  };

  const handleDiscardChanges = () => {
    onDiscardChanges();
    setHasChanges(false);
    setIsEditing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Multi-Video Editor
              </h1>
              <p className="text-slate-300 mt-2">
                AI-powered multi-video editing with intelligent chunking and
                effects
              </p>
            </div>
            <div className="flex items-center gap-4">
              <Button
                variant={isEditing ? "destructive" : "outline"}
                onClick={() => setIsEditing(!isEditing)}
                className="flex items-center gap-2"
              >
                <Edit3 size={16} />
                {isEditing ? "Exit Edit" : "Edit Mode"}
              </Button>
              {hasChanges && (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={handleDiscardChanges}
                    className="flex items-center gap-2"
                  >
                    <RotateCcw size={16} />
                    Discard
                  </Button>
                  <Button
                    onClick={handleApplyChanges}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
                  >
                    <Save size={16} />
                    Apply Changes
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Video Preview */}
          <div className="lg:col-span-2">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Video size={20} />
                  Combined Multi-Video Result
                </CardTitle>
                <CardDescription>
                  AI-generated combination of {sourceVideos.length} videos
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={() => {
                      if (videoRef.current) {
                        setDuration(videoRef.current.duration);
                      }
                    }}
                    controls
                  >
                    <source
                      src={`${
                        process.env.NEXT_PUBLIC_API_URL?.replace(
                          "/api/v1",
                          ""
                        ) || "http://localhost:8000"
                      }/api/v1/multi-video/projects/${
                        (projectStatus as any).project_id
                      }/download`}
                      type="video/mp4"
                    />
                    Your browser does not support the video tag.
                  </video>

                  {/* Video Overlay Controls */}
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                    <div className="flex items-center gap-4">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={togglePlay}
                        className="text-white hover:bg-white/20"
                      >
                        {isPlaying ? <Pause size={16} /> : <Play size={16} />}
                      </Button>

                      <div className="flex-1">
                        <input
                          type="range"
                          min="0"
                          max={duration}
                          value={currentTime}
                          onChange={handleSeek}
                          className="w-full h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer slider"
                        />
                      </div>

                      <span className="text-sm text-white">
                        {formatTime(currentTime)} / {formatTime(duration)}
                      </span>

                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={toggleMute}
                          className="text-white hover:bg-white/20"
                        >
                          {isMuted ? (
                            <VolumeX size={16} />
                          ) : (
                            <Volume2 size={16} />
                          )}
                        </Button>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={volume}
                          onChange={handleVolumeChange}
                          className="w-20 h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer slider"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Source Videos */}
            <Card className="mt-6 bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Layers size={20} />
                  Source Videos
                </CardTitle>
                <CardDescription>
                  Original videos used in the combination
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {sourceVideos.map((video, index) => (
                    <div
                      key={video.id}
                      className="relative aspect-video bg-slate-700 rounded-lg overflow-hidden"
                    >
                      <video className="w-full h-full object-cover" muted loop>
                        <source src={video.url} type="video/mp4" />
                      </video>
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                        <p className="text-xs text-white font-medium">
                          Video {index + 1}
                        </p>
                        <p className="text-xs text-slate-300">
                          {formatTime(video.duration)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Timeline and Controls */}
          <div className="space-y-6">
            {/* Timeline */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock size={20} />
                  Multi-Video Timeline
                </CardTitle>
                <CardDescription>
                  AI-generated segments with intelligent chunking
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  ref={timelineRef}
                  className="space-y-3 max-h-96 overflow-y-auto"
                >
                  {timelineSegments.map((segment, index) => {
                    const segmentId = `${segment.source_video_id}-${segment.start}-${segment.end}`;
                    const isSelected = selectedSegment === segmentId;
                    const currentSegment = getCurrentSegment();
                    const isCurrent =
                      currentSegment &&
                      currentSegment.source_video_id ===
                        segment.source_video_id &&
                      currentSegment.start === segment.start &&
                      currentSegment.end === segment.end;
                    const sourceVideo = sourceVideos.find(
                      (v) => v.id === segment.source_video_id
                    );

                    return (
                      <motion.div
                        key={`${segment.source_video_id}-${segment.start}-${segment.end}-${index}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className={`relative p-3 rounded-lg border-2 cursor-pointer transition-all ${
                          isSelected
                            ? "border-purple-500 bg-purple-500/20"
                            : isCurrent
                            ? "border-blue-500 bg-blue-500/20"
                            : "border-slate-600 bg-slate-700/50 hover:border-slate-500"
                        }`}
                        onClick={() =>
                          handleSegmentClick(
                            `${segment.source_video_id}-${segment.start}-${segment.end}`
                          )
                        }
                      >
                        {/* Segment Header */}
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                            <span className="text-sm font-medium">
                              Segment {index + 1}
                            </span>
                            {segment.effects.length > 0 && (
                              <Zap size={14} className="text-yellow-400" />
                            )}
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-slate-400">
                              {formatTime(segment.end - segment.start)}
                            </span>
                            {sourceVideo && (
                              <span className="text-xs text-slate-500">
                                (V
                                {sourceVideos.findIndex(
                                  (v) => v.id === sourceVideo.id
                                ) + 1}
                                )
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Segment Details */}
                        <div className="space-y-2">
                          {/* Effects */}
                          {segment.effects.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {segment.effects.map((effect, effectIndex) => (
                                <span
                                  key={effectIndex}
                                  className="px-2 py-1 text-xs bg-blue-500/20 text-blue-300 rounded"
                                >
                                  {effect}
                                </span>
                              ))}
                            </div>
                          )}

                          {/* Transitions */}
                          {(segment.transition_in ||
                            segment.transition_out) && (
                            <div className="flex items-center gap-2 text-xs text-slate-400">
                              <ArrowRight size={12} />
                              {segment.transition_in && (
                                <span className="bg-green-500/20 text-green-300 px-1 rounded">
                                  {segment.transition_in}
                                </span>
                              )}
                              {segment.transition_out && (
                                <span className="bg-orange-500/20 text-orange-300 px-1 rounded">
                                  {segment.transition_out}
                                </span>
                              )}
                            </div>
                          )}

                          {/* AI Recommendations */}
                          {showAIRecommendations &&
                            segment.ai_recommendations && (
                              <div className="mt-3 p-2 bg-slate-700/50 rounded border-l-2 border-purple-500">
                                <div className="flex items-center gap-2 mb-2">
                                  <Sparkles
                                    size={12}
                                    className="text-purple-400"
                                  />
                                  <span className="text-xs font-medium text-purple-300">
                                    AI Reasoning
                                  </span>
                                  <div className="flex items-center gap-1">
                                    <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                                    <span className="text-xs text-slate-400">
                                      {Math.round(
                                        segment.ai_recommendations
                                          .confidence_score * 100
                                      )}
                                      %
                                    </span>
                                  </div>
                                </div>
                                <p className="text-xs text-slate-300 leading-relaxed">
                                  {segment.ai_recommendations.segment_reasoning}
                                </p>
                              </div>
                            )}
                        </div>

                        {/* Edit Controls */}
                        {isEditing && isSelected && (
                          <div className="mt-3 pt-3 border-t border-slate-600">
                            <div className="grid grid-cols-2 gap-2">
                              <Button
                                size="sm"
                                variant="outline"
                                className="text-xs"
                                onClick={() =>
                                  handleSegmentEdit(segmentId, {
                                    speed: segment.speed === 1 ? 1.5 : 1,
                                  })
                                }
                              >
                                Speed: {segment.speed}x
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                className="text-xs"
                                onClick={() =>
                                  handleSegmentEdit(segmentId, {
                                    volume: segment.volume === 1 ? 0.5 : 1,
                                  })
                                }
                              >
                                Vol: {Math.round(segment.volume * 100)}%
                              </Button>
                            </div>
                          </div>
                        )}
                      </motion.div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* AI Insights */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lightbulb size={20} />
                  AI Insights
                </CardTitle>
                <CardDescription>
                  Intelligent analysis and recommendations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Audio Strategy */}
                  <div className="flex items-center gap-3 p-3 bg-slate-700/50 rounded-lg">
                    <Music size={16} className="text-blue-400" />
                    <div>
                      <p className="text-sm font-medium text-white">
                        Audio Strategy
                      </p>
                      <p className="text-xs text-slate-400">
                        Sequential preservation
                      </p>
                    </div>
                  </div>

                  {/* Chunking Strategy */}
                  <div className="flex items-center gap-3 p-3 bg-slate-700/50 rounded-lg">
                    <Target size={16} className="text-green-400" />
                    <div>
                      <p className="text-sm font-medium text-white">
                        Chunking Strategy
                      </p>
                      <p className="text-xs text-slate-400">
                        Scene-based intelligent splitting
                      </p>
                    </div>
                  </div>

                  {/* Effects Applied */}
                  <div className="flex items-center gap-3 p-3 bg-slate-700/50 rounded-lg">
                    <Zap size={16} className="text-yellow-400" />
                    <div>
                      <p className="text-sm font-medium text-white">
                        Effects Applied
                      </p>
                      <p className="text-xs text-slate-400">
                        {timelineSegments.reduce(
                          (total, seg) => total + seg.effects.length,
                          0
                        )}{" "}
                        effects
                      </p>
                    </div>
                  </div>

                  {/* Toggle AI Recommendations */}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setShowAIRecommendations(!showAIRecommendations)
                    }
                    className="w-full flex items-center gap-2"
                  >
                    {showAIRecommendations ? (
                      <EyeOff size={16} />
                    ) : (
                      <Eye size={16} />
                    )}
                    {showAIRecommendations ? "Hide" : "Show"} AI Recommendations
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Project Status */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings size={20} />
                  Project Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-300">Status</span>
                    <span className="text-sm font-medium text-green-400">
                      {projectStatus.status}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-300">Progress</span>
                    <span className="text-sm font-medium text-blue-400">
                      {Math.round(projectStatus.progress)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-300">Videos</span>
                    <span className="text-sm font-medium text-purple-400">
                      {projectStatus.video_count}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-300">Segments</span>
                    <span className="text-sm font-medium text-orange-400">
                      {timelineSegments.length}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Timeline Controls */}
        <div className="flex items-center justify-between p-4 bg-gray-50 border-t">
          <div className="flex items-center space-x-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsEditing(!isEditing)}
              className="flex items-center space-x-2"
            >
              <Edit3 className="w-4 h-4" />
              <span>{isEditing ? "Exit Edit" : "Edit Timeline"}</span>
            </Button>

            {/* LLM Timeline Testing */}
            {projectId && (
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={testLLMTimeline}
                  disabled={isTestingLLM}
                  className="flex items-center space-x-2"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>
                    {isTestingLLM ? "Testing..." : "Test LLM Timeline"}
                  </span>
                </Button>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={getLLMTestResult}
                  className="flex items-center space-x-2"
                >
                  <Eye className="w-4 h-4" />
                  <span>View LLM Results</span>
                </Button>
              </div>
            )}
          </div>

          {isEditing && (
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleDiscardChanges}
                disabled={!hasChanges}
              >
                <X className="w-4 h-4 mr-2" />
                Discard
              </Button>
              <Button
                onClick={handleApplyChanges}
                disabled={!hasChanges}
                className="flex items-center space-x-2"
              >
                <Save className="w-4 h-4" />
                <span>Apply Changes</span>
              </Button>
            </div>
          )}
        </div>

        {/* LLM Test Results */}
        {showLLMResults && llmTestResult && (
          <div className="p-4 bg-blue-50 border-t">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-blue-900">
                LLM Timeline Test Results
              </h3>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowLLMResults(false)}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Segments Generated</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold text-blue-600">
                    {llmTestResult.details?.segments_count || 0}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Confidence Score</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold text-green-600">
                    {(
                      (llmTestResult.details?.confidence_score || 0) * 100
                    ).toFixed(1)}
                    %
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Estimated Duration</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold text-purple-600">
                    {llmTestResult.details?.estimated_duration?.toFixed(1) || 0}
                    s
                  </p>
                </CardContent>
              </Card>
            </div>

            {llmTestResult.details?.overall_strategy && (
              <div className="mt-4">
                <h4 className="font-semibold text-blue-900 mb-2">Strategy</h4>
                <p className="text-sm text-blue-800 bg-blue-100 p-3 rounded">
                  {llmTestResult.details.overall_strategy}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
