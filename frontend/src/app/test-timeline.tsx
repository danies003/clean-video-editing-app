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
}

export default function TestTimelinePage() {
  const [segments, setSegments] = useState<TimelineSegment[]>([
    {
      id: "1",
      startTime: 0,
      endTime: 5,
      duration: 5,
      transitionIn: "fade_in",
      transitionOut: "crossfade",
      effects: ["speed_up"],
    },
    {
      id: "2",
      startTime: 5,
      endTime: 12,
      duration: 7,
      transitionIn: "crossfade",
      transitionOut: "whip_pan",
      effects: ["slow_motion"],
    },
    {
      id: "3",
      startTime: 12,
      endTime: 18,
      duration: 6,
      transitionIn: "whip_pan",
      transitionOut: "zoom_blur",
      effects: ["audio_emphasis"],
    },
  ]);

  const handleSegmentsChange = (newSegments: TimelineSegment[]) => {
    setSegments(newSegments);
    console.log("Segments updated:", newSegments);
  };

  const handlePreviewSegment = (segmentId: string) => {
    console.log("Preview segment:", segmentId);
    const segment = segments.find((s) => s.id === segmentId);
    if (segment) {
      alert(
        `Previewing segment ${segmentId}: ${segment.startTime}s - ${segment.endTime}s`
      );
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">
          Timeline Editor Test
        </h1>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Test Instructions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium mb-2">Mouse Controls</h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li>Drag segments to move them around the timeline</li>
                <li>Use the grip handles on the sides to resize segments</li>
                <li>Click on segments to select them and see details</li>
                <li>
                  Use the scissors button to split segments at the playhead
                </li>
                <li>Use the + and - buttons to zoom in/out</li>
                <li>Click on the timeline to move the playhead</li>
                <li>Use the play/pause button to animate the playhead</li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium mb-2">Keyboard Shortcuts</h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li>
                  <kbd className="px-2 py-1 bg-gray-200 rounded text-xs">
                    Space
                  </kbd>{" "}
                  - Play/Pause
                </li>
                <li>
                  <kbd className="px-2 py-1 bg-gray-200 rounded text-xs">←</kbd>{" "}
                  - Move playhead back 1s
                </li>
                <li>
                  <kbd className="px-2 py-1 bg-gray-200 rounded text-xs">→</kbd>{" "}
                  - Move playhead forward 1s
                </li>
                <li>
                  <kbd className="px-2 py-1 bg-gray-200 rounded text-xs">
                    Home
                  </kbd>{" "}
                  - Jump to start
                </li>
                <li>
                  <kbd className="px-2 py-1 bg-gray-200 rounded text-xs">
                    End
                  </kbd>{" "}
                  - Jump to end
                </li>
                <li>
                  <kbd className="px-2 py-1 bg-gray-200 rounded text-xs">
                    Delete
                  </kbd>{" "}
                  - Delete selected segment
                </li>
              </ul>
            </div>
          </div>
        </div>

        <TimelineEditor
          videoId="test-video"
          duration={20}
          segments={segments}
          onSegmentsChange={handleSegmentsChange}
          onPreviewSegment={handlePreviewSegment}
        />

        <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Current Segments</h2>
          <div className="space-y-2">
            {segments.map((segment) => (
              <div
                key={segment.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded"
              >
                <div>
                  <span className="font-medium">Segment {segment.id}:</span>
                  <span className="ml-2 text-gray-600">
                    {segment.startTime}s - {segment.endTime}s (
                    {segment.duration}s)
                  </span>
                </div>
                <div className="text-sm text-gray-500">
                  {segment.transitionIn} → {segment.transitionOut}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
