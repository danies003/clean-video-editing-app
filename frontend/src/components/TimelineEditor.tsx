"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  Scissors,
  Move,
  Trash2,
  Eye,
  EyeOff,
  Settings,
  RotateCcw,
  GripVertical,
  Plus,
  Merge,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { SegmentEditor } from "./SegmentEditor";

interface TimelineSegment {
  id: string;
  startTime: number;
  endTime: number;
  duration: number;
  transitionIn?: string;
  transitionOut?: string;
  effects?: string[];
  thumbnail?: string;
  // New: Effect customization support
  enabled?: boolean; // Segment enable/disable
  effectCustomizations?: {
    [effectName: string]: {
      enabled: boolean;
      parameters?: { [paramName: string]: any };
    };
  };
}

interface TimelineEditorProps {
  videoId: string;
  duration: number;
  segments: TimelineSegment[];
  onSegmentsChange: (segments: TimelineSegment[]) => void;
  onPreviewSegment: (segmentId: string) => void;
  className?: string;
}

interface DragState {
  isDragging: boolean;
  segmentId: string | null;
  startX: number;
  startTime: number;
  dragType: "move" | "resize-start" | "resize-end";
  originalSegment?: TimelineSegment;
}

export function TimelineEditor({
  videoId,
  duration,
  segments,
  onSegmentsChange,
  onPreviewSegment,
  className,
}: TimelineEditorProps) {
  const [dragState, setDragState] = useState<DragState>({
    isDragging: false,
    segmentId: null,
    startX: 0,
    startTime: 0,
    dragType: "move",
  });
  const [selectedSegment, setSelectedSegment] = useState<string | null>(null);
  const [showThumbnails, setShowThumbnails] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);

  // New: Segment customization state
  const [editingSegment, setEditingSegment] = useState<TimelineSegment | null>(
    null
  );
  const [segmentCustomizations, setSegmentCustomizations] = useState<{
    [segmentId: string]: {
      enabled: boolean;
      effectCustomizations: {
        [effectName: string]: {
          enabled: boolean;
          parameters?: { [paramName: string]: any };
        };
      };
    };
  }>({});

  // Convert time to pixel position with zoom
  const timeToPixels = useCallback(
    (time: number) => {
      const timelineWidth = timelineRef.current?.clientWidth || 800;
      return (time / duration) * timelineWidth * zoom;
    },
    [duration, zoom]
  );

  // Convert pixel position to time with zoom
  const pixelsToTime = useCallback(
    (pixels: number) => {
      const timelineWidth = timelineRef.current?.clientWidth || 800;
      return Math.max(
        0,
        Math.min(duration, (pixels / zoom) * (duration / timelineWidth))
      );
    },
    [duration, zoom]
  );

  // Check for segment collisions
  const checkCollisions = useCallback(
    (segmentId: string, newStart: number, newEnd: number) => {
      return segments.every((segment) => {
        if (segment.id === segmentId) return true;
        return newEnd <= segment.startTime || newStart >= segment.endTime;
      });
    },
    [segments]
  );

  // New: Segment customization functions
  const handleEditSegment = (segment: TimelineSegment) => {
    setEditingSegment(segment);
  };

  const handleSegmentChange = (updatedSegment: TimelineSegment) => {
    // Update the segments array with the modified segment
    const updatedSegments = segments.map((s) =>
      s.id === updatedSegment.id ? updatedSegment : s
    );
    onSegmentsChange(updatedSegments);
    setEditingSegment(null);
  };

  const updateSegmentEnabled = (segmentId: string, enabled: boolean) => {
    setSegmentCustomizations((prev) => ({
      ...prev,
      [segmentId]: { ...prev[segmentId], enabled },
    }));
  };

  const updateEffectEnabled = (
    segmentId: string,
    effectName: string,
    enabled: boolean
  ) => {
    setSegmentCustomizations((prev) => ({
      ...prev,
      [segmentId]: {
        ...prev[segmentId],
        effectCustomizations: {
          ...prev[segmentId]?.effectCustomizations,
          [effectName]: {
            ...prev[segmentId]?.effectCustomizations?.[effectName],
            enabled,
          },
        },
      },
    }));
  };

  const updateEffectParameter = (
    segmentId: string,
    effectName: string,
    paramName: string,
    value: any
  ) => {
    setSegmentCustomizations((prev) => ({
      ...prev,
      [segmentId]: {
        ...prev[segmentId],
        effectCustomizations: {
          ...prev[segmentId]?.effectCustomizations,
          [effectName]: {
            ...prev[segmentId]?.effectCustomizations?.[effectName],
            parameters: {
              ...prev[segmentId]?.effectCustomizations?.[effectName]
                ?.parameters,
              [paramName]: value,
            },
          },
        },
      },
    }));
  };

  // Handle mouse down on segment
  const handleSegmentMouseDown = (
    e: React.MouseEvent,
    segmentId: string,
    dragType: "move" | "resize-start" | "resize-end"
  ) => {
    e.preventDefault();
    e.stopPropagation();

    const rect = timelineRef.current?.getBoundingClientRect();
    if (!rect) return;

    const segment = segments.find((s) => s.id === segmentId);
    if (!segment) return;

    setDragState({
      isDragging: true,
      segmentId,
      startX: e.clientX - rect.left,
      startTime: pixelsToTime(e.clientX - rect.left),
      dragType,
      originalSegment: { ...segment },
    });
    setSelectedSegment(segmentId);
    setIsDragging(true);
  };

  // Handle mouse move during drag
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (
        !dragState.isDragging ||
        !dragState.segmentId ||
        !dragState.originalSegment
      )
        return;

      const rect = timelineRef.current?.getBoundingClientRect();
      if (!rect) return;

      const currentX = e.clientX - rect.left;
      const deltaX = currentX - dragState.startX;
      const deltaTime = pixelsToTime(deltaX);

      const segmentIndex = segments.findIndex(
        (s) => s.id === dragState.segmentId
      );
      if (segmentIndex === -1) return;

      const updatedSegments = [...segments];
      const segment = { ...updatedSegments[segmentIndex] };
      const original = dragState.originalSegment;

      switch (dragState.dragType) {
        case "move":
          const newStartTime = Math.max(0, original.startTime + deltaTime);
          const newEndTime = Math.min(duration, original.endTime + deltaTime);

          if (checkCollisions(segment.id, newStartTime, newEndTime)) {
            segment.startTime = newStartTime;
            segment.endTime = newEndTime;
            segment.duration = newEndTime - newStartTime;
          }
          break;

        case "resize-start":
          const newStart = Math.max(
            0,
            Math.min(original.endTime - 0.5, original.startTime + deltaTime)
          );
          if (checkCollisions(segment.id, newStart, original.endTime)) {
            segment.startTime = newStart;
            segment.duration = original.endTime - newStart;
          }
          break;

        case "resize-end":
          const newEnd = Math.max(
            original.startTime + 0.5,
            Math.min(duration, original.endTime + deltaTime)
          );
          if (checkCollisions(segment.id, original.startTime, newEnd)) {
            segment.endTime = newEnd;
            segment.duration = newEnd - original.startTime;
          }
          break;
      }

      updatedSegments[segmentIndex] = segment;
      onSegmentsChange(updatedSegments);
    },
    [
      dragState,
      segments,
      duration,
      pixelsToTime,
      checkCollisions,
      onSegmentsChange,
    ]
  );

  // Handle mouse up to end drag
  const handleMouseUp = useCallback(() => {
    setDragState({
      isDragging: false,
      segmentId: null,
      startX: 0,
      startTime: 0,
      dragType: "move",
    });
    setIsDragging(false);
  }, []);

  // Delete segment
  const deleteSegment = (segmentId: string) => {
    const updatedSegments = segments.filter((s) => s.id !== segmentId);
    onSegmentsChange(updatedSegments);
    setSelectedSegment(null);
  };

  // Split segment at current time
  const splitSegment = () => {
    const segmentIndex = segments.findIndex(
      (s) => s.startTime <= currentTime && s.endTime >= currentTime
    );

    if (segmentIndex === -1) return;

    const segment = segments[segmentIndex];
    const updatedSegments = [...segments];

    // Create two new segments
    const segment1: TimelineSegment = {
      ...segment,
      id: `${segment.id}_1`,
      endTime: currentTime,
      duration: currentTime - segment.startTime,
    };

    const segment2: TimelineSegment = {
      ...segment,
      id: `${segment.id}_2`,
      startTime: currentTime,
      duration: segment.endTime - currentTime,
    };

    updatedSegments.splice(segmentIndex, 1, segment1, segment2);
    onSegmentsChange(updatedSegments);
  };

  // Auto-playhead movement
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentTime((prev) => {
        const newTime = prev + 0.1;
        if (newTime >= duration) {
          setIsPlaying(false);
          return 0;
        }
        return newTime;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, duration]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target !== document.body) return;

      switch (e.key) {
        case " ":
          e.preventDefault();
          setIsPlaying(!isPlaying);
          break;
        case "ArrowLeft":
          e.preventDefault();
          setCurrentTime(Math.max(0, currentTime - 1));
          break;
        case "ArrowRight":
          e.preventDefault();
          setCurrentTime(Math.min(duration, currentTime + 1));
          break;
        case "Home":
          e.preventDefault();
          setCurrentTime(0);
          break;
        case "End":
          e.preventDefault();
          setCurrentTime(duration);
          break;
        case "Delete":
        case "Backspace":
          if (selectedSegment) {
            deleteSegment(selectedSegment);
          }
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isPlaying, currentTime, duration, selectedSegment]);

  // Add segment at current time
  const addSegment = () => {
    const newSegment: TimelineSegment = {
      id: `segment_${Date.now()}`,
      startTime: currentTime,
      endTime: Math.min(duration, currentTime + 3),
      duration: 3,
      transitionIn: "fade_in",
      transitionOut: "fade_out",
      effects: [],
    };

    const updatedSegments = [...segments, newSegment].sort(
      (a, b) => a.startTime - b.startTime
    );
    onSegmentsChange(updatedSegments);
    setSelectedSegment(newSegment.id);
  };

  // Merge selected segment with next segment
  const mergeWithNext = () => {
    if (!selectedSegment) return;

    const currentIndex = segments.findIndex((s) => s.id === selectedSegment);
    if (currentIndex === -1 || currentIndex === segments.length - 1) return;

    const currentSeg = segments[currentIndex];
    const nextSeg = segments[currentIndex + 1];

    const mergedSegment: TimelineSegment = {
      id: `${currentSeg.id}_merged`,
      startTime: currentSeg.startTime,
      endTime: nextSeg.endTime,
      duration: nextSeg.endTime - currentSeg.startTime,
      transitionIn: currentSeg.transitionIn,
      transitionOut: nextSeg.transitionOut,
      effects: [...(currentSeg.effects || []), ...(nextSeg.effects || [])],
    };

    const updatedSegments = [...segments];
    updatedSegments.splice(currentIndex, 2, mergedSegment);
    onSegmentsChange(updatedSegments);
    setSelectedSegment(mergedSegment.id);
  };

  // Format time for display
  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  // Generate ruler markers based on zoom
  const generateRulerMarkers = () => {
    const markers = [];
    const step = zoom >= 2 ? 1 : zoom >= 1 ? 5 : 10;
    const totalSteps = Math.ceil(duration / step);

    for (let i = 0; i <= totalSteps; i++) {
      const time = i * step;
      if (time <= duration) {
        markers.push({
          time,
          position: timeToPixels(time),
        });
      }
    }
    return markers;
  };

  return (
    <>
      <div
        className={cn("bg-white rounded-2xl p-6 shadow-lg border", className)}
      >
        {/* Timeline Controls */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentTime(0)}
            >
              <RotateCcw className="w-4 h-4" />
            </Button>

            <span className="text-sm text-gray-600">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowThumbnails(!showThumbnails)}
              title="Toggle thumbnails"
            >
              {showThumbnails ? (
                <Eye className="w-4 h-4" />
              ) : (
                <EyeOff className="w-4 h-4" />
              )}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={splitSegment}
              disabled={!selectedSegment}
              title="Split segment at playhead (S)"
            >
              <Scissors className="w-4 h-4" />
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={addSegment}
              title="Add segment at playhead"
            >
              <Move className="w-4 h-4" />
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={mergeWithNext}
              disabled={!selectedSegment}
              title="Merge with next segment"
            >
              <Settings className="w-4 h-4" />
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setZoom(Math.min(3, zoom + 0.5))}
              title="Zoom in"
            >
              +
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setZoom(Math.max(0.5, zoom - 0.5))}
              title="Zoom out"
            >
              -
            </Button>
          </div>
        </div>

        {/* Timeline Ruler */}
        <div className="relative mb-2">
          <div className="h-8 bg-gray-100 rounded flex items-center px-2 relative overflow-hidden">
            {generateRulerMarkers().map((marker) => (
              <div
                key={marker.time}
                className="absolute flex flex-col items-center"
                style={{ left: marker.position }}
              >
                <div className="w-px h-4 bg-gray-400"></div>
                <span className="text-xs text-gray-500 mt-1">
                  {formatTime(marker.time)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Timeline Track */}
        <div
          ref={timelineRef}
          className={cn(
            "relative h-32 bg-gray-50 rounded border-2 border-gray-200 cursor-pointer",
            isDragging && "cursor-grabbing"
          )}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={(e) => {
            if (isDragging) return;
            const rect = timelineRef.current?.getBoundingClientRect();
            if (rect) {
              const clickTime = pixelsToTime(e.clientX - rect.left);
              setCurrentTime(clickTime);
            }
          }}
        >
          {/* Playhead */}
          <motion.div
            ref={playheadRef}
            className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10 pointer-events-none"
            style={{ left: timeToPixels(currentTime) }}
            animate={{ left: timeToPixels(currentTime) }}
            transition={{ duration: 0.1 }}
          />

          {/* Segments */}
          <AnimatePresence>
            {segments.map((segment) => (
              <motion.div
                key={segment.id}
                className={cn(
                  "absolute h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded cursor-move",
                  "border-2 border-white shadow-md hover:shadow-lg transition-shadow",
                  selectedSegment === segment.id && "ring-2 ring-yellow-400",
                  dragState.segmentId === segment.id && "opacity-75 scale-105",
                  isDragging &&
                    dragState.segmentId !== segment.id &&
                    "pointer-events-none"
                )}
                style={{
                  left: timeToPixels(segment.startTime),
                  width: timeToPixels(segment.duration),
                }}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                onMouseDown={(e) =>
                  handleSegmentMouseDown(e, segment.id, "move")
                }
              >
                {/* Segment Content */}
                <div className="h-full flex items-center justify-between px-2 text-white text-xs">
                  <div className="flex-1 truncate">
                    {formatTime(segment.startTime)} -{" "}
                    {formatTime(segment.endTime)}
                  </div>

                  {/* Effect count badge */}
                  {segment.effects && segment.effects.length > 0 && (
                    <div className="flex items-center space-x-1 mr-1">
                      <span className="bg-white/20 px-1 rounded text-xs">
                        {segment.effects.length} effects
                      </span>
                    </div>
                  )}

                  {selectedSegment === segment.id && (
                    <div className="flex items-center space-x-1">
                      <button
                        className="p-1 hover:bg-white/20 rounded transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEditSegment(segment);
                        }}
                        title="Edit effects"
                      >
                        <Settings className="w-3 h-3" />
                      </button>
                      <button
                        className="p-1 hover:bg-white/20 rounded transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          onPreviewSegment(segment.id);
                        }}
                      >
                        <Eye className="w-3 h-3" />
                      </button>
                      <button
                        className="p-1 hover:bg-white/20 rounded transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteSegment(segment.id);
                        }}
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>

                {/* Resize Handles */}
                <div
                  className="absolute left-0 top-0 bottom-0 w-3 cursor-ew-resize bg-black/30 hover:bg-black/50 transition-colors flex items-center justify-center"
                  onMouseDown={(e) =>
                    handleSegmentMouseDown(e, segment.id, "resize-start")
                  }
                >
                  <GripVertical className="w-2 h-2 text-white" />
                </div>
                <div
                  className="absolute right-0 top-0 bottom-0 w-3 cursor-ew-resize bg-black/30 hover:bg-black/50 transition-colors flex items-center justify-center"
                  onMouseDown={(e) =>
                    handleSegmentMouseDown(e, segment.id, "resize-end")
                  }
                >
                  <GripVertical className="w-2 h-2 text-white" />
                </div>

                {/* Transition Indicators */}
                {segment.transitionIn && (
                  <div className="absolute left-0 top-0 w-5 h-5 bg-yellow-400 rounded-full -translate-x-2 -translate-y-2 flex items-center justify-center shadow-md">
                    <span className="text-xs text-black font-bold">I</span>
                  </div>
                )}
                {segment.transitionOut && (
                  <div className="absolute right-0 top-0 w-5 h-5 bg-yellow-400 rounded-full translate-x-2 -translate-y-2 flex items-center justify-center shadow-md">
                    <span className="text-xs text-black font-bold">O</span>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {/* Segment Details */}
        {selectedSegment && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-4 bg-gray-50 rounded-lg"
          >
            <h4 className="font-semibold text-gray-900 mb-2">
              Segment Details
            </h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Start:</span>{" "}
                {formatTime(
                  segments.find((s) => s.id === selectedSegment)?.startTime || 0
                )}
              </div>
              <div>
                <span className="text-gray-600">End:</span>{" "}
                {formatTime(
                  segments.find((s) => s.id === selectedSegment)?.endTime || 0
                )}
              </div>
              <div>
                <span className="text-gray-600">Duration:</span>{" "}
                {formatTime(
                  segments.find((s) => s.id === selectedSegment)?.duration || 0
                )}
              </div>
              <div>
                <span className="text-gray-600">Transition In:</span>{" "}
                {segments.find((s) => s.id === selectedSegment)?.transitionIn ||
                  "None"}
              </div>
              <div>
                <span className="text-gray-600">Transition Out:</span>{" "}
                {segments.find((s) => s.id === selectedSegment)
                  ?.transitionOut || "None"}
              </div>
              <div>
                <span className="text-gray-600">Effects:</span>{" "}
                {segments
                  .find((s) => s.id === selectedSegment)
                  ?.effects?.join(", ") || "None"}
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Segment Editor Modal */}
      {editingSegment && (
        <SegmentEditor
          segment={{
            ...editingSegment,
            effectCustomizations: editingSegment.effectCustomizations
              ? Object.fromEntries(
                  Object.entries(editingSegment.effectCustomizations).map(
                    ([key, value]) => [
                      key,
                      {
                        ...value,
                        parameters: value.parameters || {},
                      },
                    ]
                  )
                )
              : undefined,
          }}
          onSegmentChange={handleSegmentChange}
          onClose={() => setEditingSegment(null)}
        />
      )}
    </>
  );
}
