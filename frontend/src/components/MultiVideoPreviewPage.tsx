"use client";

import React, { useState } from "react";
import VideoTimelineEditor from "./VideoTimelineEditor";

interface MultiVideoSegment {
  start: number; // ✅ Same as EditDecisionSegment
  end: number; // ✅ Same as EditDecisionSegment
  source_video_id: string;
  video_url?: string; // Add video URL for original video streaming
  stream_url?: string; // Add stream URL for processed video streaming
  effects: string[]; // Will be converted to tags
  transition_in?: string; // ✅ Same as EditDecisionSegment
  transition_out?: string; // ✅ Same as EditDecisionSegment
  speed: number; // ✅ Same as EditDecisionSegment
  volume: number;
  effectCustomizations?: { [key: string]: any }; // ✅ Same as EditDecisionSegment
  ai_recommendations?: {
    segment_reasoning: string;
    transition_reasoning: string;
    effects_reasoning: string;
    arrangement_reasoning: string;
    confidence_score: number;
    alternative_suggestions: string[];
  };
}

interface MultiVideoPreviewPageProps {
  sourceVideos: Array<{
    id: string;
    url: string;
    name: string;
    duration: number;
  }>;
  timelineSegments: MultiVideoSegment[];
  outputVideoUrl?: string;
  llmSuggestions?: any[];
  projectId?: string; // Add project ID for LLM testing
  onUpdateSegment: (
    segmentIndex: number,
    updates: Partial<MultiVideoSegment>
  ) => void;
  onBack: () => void;
  onProceed: () => void;
}

const MultiVideoPreviewPage: React.FC<MultiVideoPreviewPageProps> = ({
  sourceVideos,
  timelineSegments,
  outputVideoUrl,
  llmSuggestions = [],
  projectId, // Add project ID
  onUpdateSegment,
  onBack,
  onProceed,
}) => {
  // Calculate actual video duration considering speed effects
  const calculateActualDuration = () => {
    let actualDuration = 0;
    for (const segment of timelineSegments) {
      const segmentDuration = segment.end - segment.start;
      const speed = segment.speed || 1.0;
      // Speed effects make segments shorter (higher speed = shorter duration)
      const actualSegmentDuration = segmentDuration / speed;
      actualDuration += actualSegmentDuration;
    }
    return actualDuration;
  };

  // Convert multi-video segments to the format expected by VideoTimelineEditor
  // Use original timeline positions since LLM now generates sequential timeline positions
  const convertedSegments = timelineSegments.map((segment, index) => {
    // Use original timeline positions directly (LLM now generates sequential positions)
    const sequentialStart = segment.start;
    const sequentialEnd = segment.end;

    // Calculate actual segment duration for logging
    const segmentDuration = segment.end - segment.start;
    const speed = segment.speed || 1.0;
    const actualSegmentDuration = segmentDuration / speed; // Speed effects make segments shorter

    console.log(
      `🔧 [MultiVideoPreviewPage] Segment ${index + 1}: Timeline ${
        segment.start
      }-${
        segment.end
      }s (speed: ${speed}x, actual: ${actualSegmentDuration.toFixed(2)}s)`
    );

    return {
      start: sequentialStart, // Use original timeline position
      end: sequentialEnd, // Use original timeline position
      transition: segment.transition_in,
      transition_duration: 0.5,
      tags: segment.effects, // ✅ Convert effects to tags for compatibility
      speed: segment.speed, // ✅ Exact same field name
      transition_in: segment.transition_in, // ✅ Exact same field name
      transition_out: segment.transition_out, // ✅ Exact same field name
      enabled: true,
      effectCustomizations: segment.effectCustomizations, // ✅ Exact same field name
    };
  });

  // Calculate total duration from the converted sequential segments
  const totalDuration =
    convertedSegments.length > 0
      ? convertedSegments[convertedSegments.length - 1].end
      : 0;

  // Debug logging for segment visibility
  console.log("🔧 [MultiVideoPreviewPage] Segment visibility analysis:", {
    totalSegments: timelineSegments.length,
    convertedSegments: convertedSegments.length,
    totalDuration,
    segments: convertedSegments.map((seg, idx) => ({
      index: idx + 1,
      start: seg.start,
      end: seg.end,
      duration: seg.end - seg.start,
      widthPercentage: ((seg.end - seg.start) / totalDuration) * 100,
      effects: seg.tags?.length || 0,
    })),
  });

  // Check if segments have been re-arranged by LLM
  const segmentOrdering = (timelineSegments as any).segment_ordering || {};
  const hasRearrangement = segmentOrdering.has_rearrangement || false;
  const optimalSequence = segmentOrdering.optimal_sequence || [];
  const rearrangementReasoning = segmentOrdering.rearrangement_reasoning || "";

  // State for toggling between original and optimal order
  const [useOptimalOrder, setUseOptimalOrder] = useState(true);

  console.log("🔧 [MultiVideoPreviewPage] Using totalDuration:", totalDuration);
  console.log(
    "🔧 [MultiVideoPreviewPage] Timeline segments with speeds:",
    timelineSegments.map((s) => ({
      start: s.start,
      end: s.end,
      speed: s.speed || 1.0,
      duration: s.end - s.start,
      actualDuration: (s.end - s.start) / (s.speed || 1.0),
    }))
  );
  console.log(
    "🔧 [MultiVideoPreviewPage] Segment end times:",
    timelineSegments.map((s) => s.end)
  );
  console.log(
    "🔧 [MultiVideoPreviewPage] Segment start times:",
    timelineSegments.map((s) => s.start)
  );

  // Debug logging
  console.log(
    "🔧 [MultiVideoPreviewPage] timelineSegments length:",
    timelineSegments.length
  );
  console.log(
    "🔧 [MultiVideoPreviewPage] convertedSegments length:",
    convertedSegments.length
  );
  console.log(
    "🔧 [MultiVideoPreviewPage] Original timelineSegments:",
    timelineSegments
  );
  console.log(
    "🔧 [MultiVideoPreviewPage] Re-timed convertedSegments:",
    convertedSegments
  );
  console.log(
    "🔧 [MultiVideoPreviewPage] Total sequential duration:",
    convertedSegments.length > 0
      ? convertedSegments[convertedSegments.length - 1].end
      : 0
  );
  console.log("🔧 [MultiVideoPreviewPage] Expected vs Actual duration:", {
    timelineDuration: 20.0,
    calculatedDuration: totalDuration,
    difference: 20.0 - totalDuration,
  });
  console.log("🔧 [MultiVideoPreviewPage] Segment ordering info:", {
    hasRearrangement,
    optimalSequence,
    rearrangementReasoning,
    segmentOrdering,
    useOptimalOrder,
  });
  console.log("🔧 [MultiVideoPreviewPage] LLM suggestions:", llmSuggestions);

  const handleSegmentUpdate = (segmentIndex: number, updatedSegment: any) => {
    // Convert back to multi-video segment format
    const updates: Partial<MultiVideoSegment> = {
      start: updatedSegment.start,
      end: updatedSegment.end,
      effects: updatedSegment.tags || [], // ✅ Convert tags back to effects
      transition_in: updatedSegment.transition_in,
      transition_out: updatedSegment.transition_out,
      speed: updatedSegment.speed || 1.0,
      effectCustomizations: updatedSegment.effectCustomizations,
    };
    onUpdateSegment(segmentIndex, updates);
  };

  return (
    <div className="w-full h-full flex flex-col">
      {/* LLM Rearrangement Indicator */}
      {hasRearrangement && (
        <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-blue-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3 flex-1">
              <h3 className="text-sm font-medium text-blue-800">
                🧠 AI-Optimized Segment Order
              </h3>
              <div className="mt-2 text-sm text-blue-700">
                <p>
                  The AI has re-arranged the video segments for optimal flow:
                </p>
                <p className="mt-1 font-mono text-xs">
                  Original:{" "}
                  {segmentOrdering.original_video_order?.join(" → ") ||
                    "Unknown"}
                </p>
                <p className="mt-1 font-mono text-xs">
                  Optimal: {optimalSequence.join(" → ") || "Unknown"}
                </p>
                {rearrangementReasoning && (
                  <p className="mt-2 italic">"{rearrangementReasoning}"</p>
                )}
              </div>
            </div>
            <div className="flex-shrink-0">
              <button
                onClick={() => setUseOptimalOrder(!useOptimalOrder)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  useOptimalOrder
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                }`}
              >
                {useOptimalOrder ? "Optimal Order" : "Original Order"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Video Timeline Editor - Full Width */}
      <div className="flex-1">
        <VideoTimelineEditor
          videoFile={undefined} // Remove empty file - let component handle video loading
          convertedVideoUrl={
            timelineSegments[0]?.stream_url
              ? `http://localhost:8000${timelineSegments[0].stream_url}`
              : timelineSegments[0]?.video_url
              ? `http://localhost:8000${timelineSegments[0].video_url}`
              : sourceVideos[0]?.url
          }
          segments={convertedSegments}
          onSegmentUpdate={handleSegmentUpdate}
          onPlaybackTimeChange={(time: number) => {
            // Handle playback time changes
            console.log("Playback time changed:", time);
          }}
          onBack={onBack}
          onProceed={onProceed}
          customDuration={totalDuration}
          llmSuggestions={llmSuggestions}
          projectId={projectId} // Pass projectId to VideoTimelineEditor
        />
      </div>
    </div>
  );
};

export default MultiVideoPreviewPage;
