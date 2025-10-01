"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import VideoTimelineEditor from "./VideoTimelineEditor";

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
}

interface PreviewPageProps {
  videoFile: File;
  convertedVideoUrl?: string;
  editDecision: EditDecisionMap;
  onProceed: () => void;
  onAdjust: (
    segmentIndex: number,
    adjustments: Partial<EditDecisionSegment>
  ) => void;
  onBack: () => void;
  llmSuggestions?: LLMSuggestion[];
}

const PreviewPage: React.FC<PreviewPageProps> = ({
  videoFile,
  convertedVideoUrl,
  editDecision,
  onProceed,
  onAdjust,
  onBack,
  llmSuggestions = [],
}) => {
  const [currentPlaybackTime, setCurrentPlaybackTime] = useState(0);

  const handleSegmentUpdate = (
    segmentIndex: number,
    segment: EditDecisionSegment
  ) => {
    onAdjust(segmentIndex, segment);
  };

  const handlePlaybackTimeChange = (time: number) => {
    setCurrentPlaybackTime(time);
  };

  return (
    <div className="w-full h-full flex flex-col">
      {/* Video Timeline Editor with LLM Suggestions - Full Width */}
      <div className="flex-1">
        <VideoTimelineEditor
          videoFile={videoFile}
          convertedVideoUrl={convertedVideoUrl}
          segments={editDecision.segments}
          onSegmentUpdate={handleSegmentUpdate}
          onPlaybackTimeChange={handlePlaybackTimeChange}
          llmSuggestions={llmSuggestions}
          onBack={onBack}
          onProceed={onProceed}
        />
      </div>
    </div>
  );
};

export default PreviewPage;
