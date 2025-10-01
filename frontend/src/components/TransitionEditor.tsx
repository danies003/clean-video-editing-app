"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Clock, Zap, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

interface TransitionEditorProps {
  transitionIndex: number;
  segments: any[];
  onSegmentUpdate: (segmentIndex: number, segment: any) => void;
  onClose: () => void;
}

const AVAILABLE_TRANSITIONS = [
  { value: "none", label: "None", description: "No transition effect" },
  {
    value: "cross_dissolve",
    label: "Cross Dissolve",
    description: "Smooth fade between segments",
  },
  { value: "fade_in", label: "Fade In", description: "Fade in from black" },
  { value: "fade_out", label: "Fade Out", description: "Fade out to black" },
  { value: "slide", label: "Slide", description: "Slide transition effect" },
  { value: "zoom", label: "Zoom", description: "Zoom in/out transition" },
  { value: "whip_pan", label: "Whip Pan", description: "Fast pan transition" },
  { value: "spin", label: "Spin", description: "Rotational transition" },
  { value: "glitch", label: "Glitch", description: "Digital glitch effect" },
];

export function TransitionEditor({
  transitionIndex,
  segments,
  onSegmentUpdate,
  onClose,
}: TransitionEditorProps) {
  const [localTransitionType, setLocalTransitionType] =
    useState<string>("none");
  const [localDuration, setLocalDuration] = useState<number>(0.5);
  const [localEnabled, setLocalEnabled] = useState<boolean>(true);

  const prevSegment = segments[transitionIndex - 1];
  const currentSegment = segments[transitionIndex];

  useEffect(() => {
    if (prevSegment) {
      setLocalTransitionType(prevSegment.transition_out || "none");
      setLocalDuration(prevSegment.transition_duration || 0.5);
      setLocalEnabled(prevSegment.transition_out !== "none");
    }
  }, [prevSegment]);

  const handleSave = () => {
    if (prevSegment) {
      const updatedPrevSegment = {
        ...prevSegment,
        transition_out: localEnabled ? localTransitionType : "none",
        transition_duration: localDuration,
      };
      onSegmentUpdate(transitionIndex - 1, updatedPrevSegment);
    }
    onClose();
  };

  const handleCancel = () => {
    onClose();
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins}:${secs.toString().padStart(2, "0")}.${ms
      .toString()
      .padStart(2, "0")}`;
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[85vh] overflow-hidden flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-8 border-b border-gray-200 bg-gradient-to-r from-purple-50 to-blue-50">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Edit Transition
              </h2>
              <p className="text-lg text-gray-600 flex items-center">
                Segment {transitionIndex}{" "}
                <ArrowRight className="w-4 h-4 mx-2" /> Segment{" "}
                {transitionIndex + 1}
                <span className="text-gray-400 mx-2">•</span>
                {formatTime(prevSegment?.start || 0)} -{" "}
                {formatTime(currentSegment?.end || 0)}
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-3 hover:bg-gray-100 rounded-full transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          <div className="flex-1 p-8 overflow-y-auto">
            <div className="space-y-8">
              {/* Enable/Disable Toggle */}
              <div className="flex items-center justify-between p-6 bg-gray-50 rounded-xl border">
                <div className="flex items-center space-x-3">
                  <Zap className="w-6 h-6 text-purple-600" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      Enable Transition
                    </h3>
                    <p className="text-sm text-gray-600">
                      Toggle transition effect between segments
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setLocalEnabled(!localEnabled)}
                  className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                    localEnabled ? "bg-purple-600" : "bg-gray-300"
                  }`}
                >
                  <span
                    className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform shadow-md ${
                      localEnabled ? "translate-x-7" : "translate-x-1"
                    }`}
                  />
                </button>
              </div>

              {/* Transition Type Selection */}
              {localEnabled && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <Zap className="w-5 h-5 text-purple-600" />
                    <h3 className="text-lg font-semibold text-gray-900">
                      Transition Type
                    </h3>
                  </div>
                  <div className="grid grid-cols-1 gap-3">
                    {AVAILABLE_TRANSITIONS.filter(
                      (t) => t.value !== "none"
                    ).map((transition) => (
                      <div
                        key={transition.value}
                        className={`border-2 rounded-xl p-4 cursor-pointer transition-all duration-200 ${
                          localTransitionType === transition.value
                            ? "border-purple-500 bg-purple-50 shadow-lg"
                            : "border-gray-200 hover:border-gray-300 hover:shadow-md"
                        }`}
                        onClick={() => setLocalTransitionType(transition.value)}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="font-semibold text-gray-900">
                              {transition.label}
                            </h4>
                            <p className="text-sm text-gray-600 mt-1">
                              {transition.description}
                            </p>
                          </div>
                          {localTransitionType === transition.value && (
                            <div className="w-6 h-6 bg-purple-600 rounded-full flex items-center justify-center">
                              <div className="w-2 h-2 bg-white rounded-full"></div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Transition Duration */}
              {localEnabled && localTransitionType !== "none" && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <Clock className="w-5 h-5 text-purple-600" />
                    <h3 className="text-lg font-semibold text-gray-900">
                      Transition Duration
                    </h3>
                  </div>
                  <div className="space-y-4 p-6 bg-gray-50 rounded-xl border">
                    <div className="flex justify-between items-center">
                      <div>
                        <label className="text-lg font-semibold text-gray-900">
                          Duration
                        </label>
                        <p className="text-sm text-gray-600 mt-1">
                          How long the transition effect lasts
                        </p>
                      </div>
                      <div className="text-right">
                        <span className="text-2xl font-bold text-purple-600">
                          {localDuration.toFixed(2)}s
                        </span>
                        <div className="text-xs text-gray-500">Duration</div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <input
                        type="range"
                        min={0.1}
                        max={2.0}
                        step={0.1}
                        value={localDuration}
                        onChange={(e) =>
                          setLocalDuration(parseFloat(e.target.value))
                        }
                        className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>0.1s</span>
                        <span>2.0s</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Preview */}
              {localEnabled && localTransitionType !== "none" && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Preview
                  </h3>
                  <div className="bg-gray-100 rounded-lg p-6 text-center">
                    <div className="text-lg font-medium text-gray-900 mb-2">
                      {
                        AVAILABLE_TRANSITIONS.find(
                          (t) => t.value === localTransitionType
                        )?.label
                      }
                    </div>
                    <div className="text-sm text-gray-600">
                      Duration: {localDuration.toFixed(2)}s
                    </div>
                    <div className="text-xs text-gray-500 mt-2">
                      Between segments {transitionIndex} and{" "}
                      {transitionIndex + 1}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="flex justify-end space-x-3 p-8 border-t border-gray-200">
            <Button
              variant="outline"
              onClick={handleCancel}
              className="px-6 py-2"
            >
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              className="px-6 py-2 bg-purple-600 hover:bg-purple-700"
            >
              Apply Changes
            </Button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
