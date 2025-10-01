"use client";

import React, { useState } from "react";
import { TimelineEditor } from "@/components/TimelineEditor";

interface TimelineSegment {
  id: string;
  startTime: number;
  endTime: number;
  duration: number;
  transitionIn?: string;
  transitionOut?: string;
  effects?: string[];
  thumbnail?: string;
  enabled?: boolean;
  effectCustomizations?: {
    [effectName: string]: {
      enabled: boolean;
      parameters?: { [paramName: string]: any };
    };
  };
}

export default function TestSegmentEditorPage() {
  const [segments, setSegments] = useState<TimelineSegment[]>([
    {
      id: "1",
      startTime: 0,
      endTime: 5,
      duration: 5,
      effects: ["color_grade", "visual_effect"],
      transitionIn: "fade_in",
      transitionOut: "fade_out",
      enabled: true,
    },
    {
      id: "2",
      startTime: 5,
      endTime: 10,
      duration: 5,
      effects: ["speed", "audio"],
      transitionIn: "cross_dissolve",
      transitionOut: "whip_pan",
      enabled: true,
    },
    {
      id: "3",
      startTime: 10,
      endTime: 15,
      duration: 5,
      effects: ["crop", "color_grade"],
      transitionIn: "zoom_in",
      transitionOut: "zoom_out",
      enabled: true,
    },
  ]);

  const handleSegmentsChange = (newSegments: TimelineSegment[]) => {
    setSegments(newSegments);
    console.log("Segments updated:", newSegments);
  };

  const handlePreviewSegment = (segmentId: string) => {
    console.log("Preview segment:", segmentId);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">
          Segment Editor Test
        </h1>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">
            Timeline with Effect Customization
          </h2>

          <TimelineEditor
            videoId="test-video"
            duration={15}
            segments={segments}
            onSegmentsChange={handleSegmentsChange}
            onPreviewSegment={handlePreviewSegment}
          />
        </div>

        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Current Segments Data</h2>
          <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
            {JSON.stringify(segments, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}
