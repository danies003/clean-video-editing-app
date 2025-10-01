"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Settings,
  Play,
  Pause,
  Eye,
  EyeOff,
  Trash2,
  Plus,
  Minus,
  RotateCcw,
  Palette,
  Zap,
  Volume2,
  Clock,
  Sliders,
  X,
  Video,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface EffectParameter {
  name: string;
  type: "number" | "boolean" | "select" | "color" | "range";
  value: any;
  minValue?: number;
  maxValue?: number;
  step?: number;
  options?: string[];
  label: string;
  description: string;
}

interface Effect {
  id: string;
  name: string;
  type: string;
  enabled: boolean;
  parameters: EffectParameter[];
  startTime: number;
  endTime: number;
  duration: number;
}

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
      parameters: { [paramName: string]: any };
    };
  };
}

interface SegmentEditorProps {
  segment: TimelineSegment;
  onSegmentChange: (segment: TimelineSegment) => void;
  onClose: () => void;
  className?: string;
}

// Available effects with their parameters
const AVAILABLE_EFFECTS: Record<
  string,
  {
    name: string;
    icon: React.ComponentType<any>;
    parameters: EffectParameter[];
  }
> = {
  color_grade: {
    name: "Color Grade",
    icon: Palette,
    parameters: [
      {
        name: "brightness",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 2.0,
        step: 0.1,
        label: "Brightness",
        description: "Adjust video brightness",
      },
      {
        name: "contrast",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 2.0,
        step: 0.1,
        label: "Contrast",
        description: "Adjust video contrast",
      },
      {
        name: "saturation",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 2.0,
        step: 0.1,
        label: "Saturation",
        description: "Adjust color saturation",
      },
    ],
  },
  color_grading: {
    name: "Color Grading",
    icon: Palette,
    parameters: [
      {
        name: "brightness",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 2.0,
        step: 0.1,
        label: "Brightness",
        description: "Adjust video brightness",
      },
      {
        name: "contrast",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 2.0,
        step: 0.1,
        label: "Contrast",
        description: "Adjust video contrast",
      },
      {
        name: "saturation",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 2.0,
        step: 0.1,
        label: "Saturation",
        description: "Adjust color saturation",
      },
    ],
  },
  beat_sync: {
    name: "Beat Sync",
    icon: Zap,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Beat synchronization intensity",
      },
      {
        name: "sensitivity",
        type: "range" as const,
        value: 0.5,
        minValue: 0.1,
        maxValue: 1.0,
        step: 0.1,
        label: "Sensitivity",
        description: "Beat detection sensitivity",
      },
    ],
  },
  motion_blur: {
    name: "Motion Blur",
    icon: Zap,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Motion blur intensity",
      },
      {
        name: "duration",
        type: "range" as const,
        value: 1.0,
        minValue: 0.1,
        maxValue: 5.0,
        step: 0.1,
        label: "Duration",
        description: "Blur duration in seconds",
      },
    ],
  },
  frequency_visualizer: {
    name: "Frequency Visualizer",
    icon: Volume2,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Visualization intensity",
      },
      {
        name: "frequency_range",
        type: "range" as const,
        value: 0.5,
        minValue: 0.1,
        maxValue: 1.0,
        step: 0.1,
        label: "Frequency Range",
        description: "Audio frequency range to visualize",
      },
    ],
  },
  audio_pulse: {
    name: "Audio Pulse",
    icon: Volume2,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Audio pulse intensity",
      },
      {
        name: "threshold",
        type: "range" as const,
        value: 0.5,
        minValue: 0.1,
        maxValue: 1.0,
        step: 0.1,
        label: "Threshold",
        description: "Audio threshold for pulse detection",
      },
    ],
  },
  cinematic: {
    name: "Cinematic",
    icon: Video,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Cinematic effect intensity",
      },
      {
        name: "style",
        type: "select" as const,
        value: "dramatic",
        options: ["dramatic", "romantic", "action", "mystery"],
        label: "Style",
        description: "Cinematic style",
      },
    ],
  },
  cyberpunk: {
    name: "Cyberpunk",
    icon: Zap,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Cyberpunk effect intensity",
      },
      {
        name: "neon_glow",
        type: "range" as const,
        value: 0.5,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.1,
        label: "Neon Glow",
        description: "Neon glow intensity",
      },
    ],
  },
  glitch: {
    name: "Glitch",
    icon: Zap,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Glitch effect intensity",
      },
      {
        name: "frequency",
        type: "range" as const,
        value: 0.5,
        minValue: 0.1,
        maxValue: 1.0,
        step: 0.1,
        label: "Frequency",
        description: "Glitch frequency",
      },
    ],
  },
  high_contrast: {
    name: "High Contrast",
    icon: Palette,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Contrast intensity",
      },
      {
        name: "threshold",
        type: "range" as const,
        value: 0.5,
        minValue: 0.1,
        maxValue: 1.0,
        step: 0.1,
        label: "Threshold",
        description: "Contrast threshold",
      },
    ],
  },
  vintage: {
    name: "Vintage",
    icon: Palette,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Vintage effect intensity",
      },
      {
        name: "era",
        type: "select" as const,
        value: "70s",
        options: ["50s", "60s", "70s", "80s", "90s"],
        label: "Era",
        description: "Vintage era style",
      },
    ],
  },
  film_noir: {
    name: "Film Noir",
    icon: Palette,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Film noir intensity",
      },
      {
        name: "shadow_depth",
        type: "range" as const,
        value: 0.5,
        minValue: 0.1,
        maxValue: 1.0,
        step: 0.1,
        label: "Shadow Depth",
        description: "Shadow depth intensity",
      },
    ],
  },
  twirl: {
    name: "Twirl",
    icon: RotateCcw,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Twirl effect intensity",
      },
      {
        name: "speed",
        type: "range" as const,
        value: 1.0,
        minValue: 0.1,
        maxValue: 3.0,
        step: 0.1,
        label: "Speed",
        description: "Twirl rotation speed",
      },
    ],
  },
  warp: {
    name: "Warp",
    icon: Zap,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Warp effect intensity",
      },
      {
        name: "direction",
        type: "select" as const,
        value: "horizontal",
        options: ["horizontal", "vertical", "radial"],
        label: "Direction",
        description: "Warp direction",
      },
    ],
  },
  motion_trail: {
    name: "Motion Trail",
    icon: Zap,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Motion trail intensity",
      },
      {
        name: "trail_length",
        type: "range" as const,
        value: 0.5,
        minValue: 0.1,
        maxValue: 1.0,
        step: 0.1,
        label: "Trail Length",
        description: "Length of motion trail",
      },
    ],
  },
  duotone: {
    name: "Duotone",
    icon: Palette,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Duotone effect intensity",
      },
      {
        name: "color1",
        type: "select" as const,
        value: "blue",
        options: ["blue", "red", "green", "purple", "orange"],
        label: "Primary Color",
        description: "First duotone color",
      },
      {
        name: "color2",
        type: "select" as const,
        value: "orange",
        options: ["blue", "red", "green", "purple", "orange"],
        label: "Secondary Color",
        description: "Second duotone color",
      },
    ],
  },
  visual_effect: {
    name: "Visual Effect",
    icon: Zap,
    parameters: [
      {
        name: "intensity",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.05,
        label: "Intensity",
        description: "Effect intensity",
      },
      {
        name: "duration",
        type: "range" as const,
        value: 1.0,
        minValue: 0.1,
        maxValue: 5.0,
        step: 0.1,
        label: "Duration",
        description: "Effect duration in seconds",
      },
    ],
  },
  speed: {
    name: "Speed",
    icon: Clock,
    parameters: [
      {
        name: "speed_factor",
        type: "range" as const,
        value: 1.0,
        minValue: 0.1,
        maxValue: 3.0,
        step: 0.1,
        label: "Speed Factor",
        description: "Playback speed multiplier",
      },
    ],
  },
  speed_up: {
    name: "Speed Up",
    icon: Clock,
    parameters: [
      {
        name: "speed_factor",
        type: "range" as const,
        value: 2.0,
        minValue: 1.1,
        maxValue: 5.0,
        step: 0.1,
        label: "Speed Factor",
        description: "Playback speed multiplier (faster)",
      },
    ],
  },
  slow_motion: {
    name: "Slow Motion",
    icon: Clock,
    parameters: [
      {
        name: "speed_factor",
        type: "range" as const,
        value: 0.5,
        minValue: 0.1,
        maxValue: 0.9,
        step: 0.1,
        label: "Speed Factor",
        description: "Playback speed multiplier (slower)",
      },
    ],
  },
  audio: {
    name: "Audio",
    icon: Volume2,
    parameters: [
      {
        name: "volume",
        type: "range" as const,
        value: 1.0,
        minValue: 0.0,
        maxValue: 2.0,
        step: 0.1,
        label: "Volume",
        description: "Audio volume level",
      },
    ],
  },
  crop: {
    name: "Crop",
    icon: Sliders,
    parameters: [
      {
        name: "crop_x",
        type: "range" as const,
        value: 0.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.01,
        label: "Crop X",
        description: "Horizontal crop position",
      },
      {
        name: "crop_y",
        type: "range" as const,
        value: 0.0,
        minValue: 0.0,
        maxValue: 1.0,
        step: 0.01,
        label: "Crop Y",
        description: "Vertical crop position",
      },
    ],
  },
};

// Available transitions
const AVAILABLE_TRANSITIONS = [
  { value: "none", label: "None" },
  { value: "cross_dissolve", label: "Cross Dissolve" },
  { value: "fade_in", label: "Fade In" },
  { value: "fade_out", label: "Fade Out" },
  { value: "slide", label: "Slide" },
  { value: "zoom", label: "Zoom" },
  { value: "whip_pan", label: "Whip Pan" },
  { value: "spin", label: "Spin" },
  { value: "glitch", label: "Glitch" },
];

export function SegmentEditor({
  segment,
  onSegmentChange,
  onClose,
  className,
}: SegmentEditorProps) {
  const [localSegment, setLocalSegment] = useState<TimelineSegment>(segment);
  const [selectedEffect, setSelectedEffect] = useState<string | null>(null);

  // Initialize effect customizations if they don't exist
  useEffect(() => {
    if (!localSegment.effectCustomizations) {
      const initialCustomizations: {
        [key: string]: { enabled: boolean; parameters: { [key: string]: any } };
      } = {};

      // Initialize all available effects as disabled by default
      Object.keys(AVAILABLE_EFFECTS).forEach((effectName) => {
        const effectConfig = AVAILABLE_EFFECTS[effectName];
        if (effectConfig) {
          initialCustomizations[effectName] = {
            enabled: false, // Start with all effects disabled
            parameters: effectConfig.parameters.reduce((acc, param) => {
              acc[param.name] = param.value;
              return acc;
            }, {} as { [key: string]: any }),
          };
        }
      });

      // Enable effects that are already in the segment's effects list
      if (localSegment.effects) {
        localSegment.effects.forEach((effectName) => {
          if (initialCustomizations[effectName]) {
            initialCustomizations[effectName].enabled = true;
          }
        });
      }

      setLocalSegment((prev) => ({
        ...prev,
        effectCustomizations: initialCustomizations,
      }));
    }
  }, [localSegment.effects, localSegment.effectCustomizations]);

  const updateSegment = (updates: Partial<TimelineSegment>) => {
    console.log(`[DEBUG] Updating segment:`, updates);
    setLocalSegment((prev) => {
      const updated = { ...prev, ...updates };
      console.log(`[DEBUG] Updated localSegment:`, updated);
      return updated;
    });
  };

  const updateEffectEnabled = (effectName: string, enabled: boolean) => {
    console.log(`[DEBUG] Updating effect ${effectName} enabled: ${enabled}`);
    setLocalSegment((prev) => {
      const updated = {
        ...prev,
        effectCustomizations: {
          ...prev.effectCustomizations,
          [effectName]: {
            enabled: enabled,
            parameters:
              prev.effectCustomizations?.[effectName]?.parameters || {},
          },
        },
      };
      console.log(`[DEBUG] Updated localSegment:`, updated);
      return updated;
    });
  };

  const updateEffectParameter = (
    effectName: string,
    paramName: string,
    value: any
  ) => {
    setLocalSegment((prev) => ({
      ...prev,
      effectCustomizations: {
        ...prev.effectCustomizations,
        [effectName]: {
          enabled: prev.effectCustomizations?.[effectName]?.enabled ?? true,
          parameters: {
            ...prev.effectCustomizations?.[effectName]?.parameters,
            [paramName]: value,
          },
        },
      },
    }));
  };

  const handleSave = () => {
    console.log("Save button clicked! Saving segment:", localSegment);
    console.log("Save button clicked! Segment enabled:", localSegment.enabled);
    console.log(
      "Save button clicked! Effect customizations:",
      localSegment.effectCustomizations
    );

    // Create a list of only enabled effects
    const enabledEffects = Object.keys(localSegment.effectCustomizations || {})
      .filter(
        (effectName) => localSegment.effectCustomizations?.[effectName]?.enabled
      )
      .filter((effectName) => AVAILABLE_EFFECTS[effectName]); // Only include valid effects

    const updatedSegment = {
      ...localSegment,
      effects: enabledEffects,
    };

    onSegmentChange(updatedSegment);
  };

  const handleCancel = () => {
    onClose();
  };

  const renderParameterInput = (param: EffectParameter, effectName: string) => {
    const currentValue =
      localSegment.effectCustomizations?.[effectName]?.parameters?.[
        param.name
      ] ?? param.value;

    switch (param.type) {
      case "range":
        return (
          <div
            key={param.name}
            className="space-y-4 p-6 bg-gray-50 rounded-xl border"
          >
            <div className="flex justify-between items-center">
              <div>
                <label className="text-lg font-semibold text-gray-900">
                  {param.label}
                </label>
                <p className="text-sm text-gray-600 mt-1">
                  {param.description}
                </p>
              </div>
              <div className="text-right">
                <span className="text-2xl font-bold text-blue-600">
                  {currentValue}
                </span>
                <div className="text-xs text-gray-500">Current Value</div>
              </div>
            </div>
            <div className="space-y-2">
              <input
                type="range"
                min={param.minValue}
                max={param.maxValue}
                step={param.step}
                value={currentValue}
                onChange={(e) =>
                  updateEffectParameter(
                    effectName,
                    param.name,
                    parseFloat(e.target.value)
                  )
                }
                className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>{param.minValue}</span>
                <span>{param.maxValue}</span>
              </div>
            </div>
          </div>
        );

      case "number":
        return (
          <div
            key={param.name}
            className="space-y-4 p-6 bg-gray-50 rounded-xl border"
          >
            <div>
              <label className="text-lg font-semibold text-gray-900">
                {param.label}
              </label>
              <p className="text-sm text-gray-600 mt-1">{param.description}</p>
            </div>
            <input
              type="number"
              min={param.minValue}
              max={param.maxValue}
              step={param.step}
              value={currentValue}
              onChange={(e) =>
                updateEffectParameter(
                  effectName,
                  param.name,
                  parseFloat(e.target.value)
                )
              }
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
            />
          </div>
        );

      case "boolean":
        return (
          <div
            key={param.name}
            className="flex items-center justify-between p-6 bg-gray-50 rounded-xl border"
          >
            <div>
              <label className="text-lg font-semibold text-gray-900">
                {param.label}
              </label>
              <p className="text-sm text-gray-600 mt-1">{param.description}</p>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={currentValue}
                onChange={(e) =>
                  updateEffectParameter(
                    effectName,
                    param.name,
                    e.target.checked
                  )
                }
                className="w-6 h-6 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="ml-3 text-sm font-medium text-gray-700">
                {currentValue ? "Enabled" : "Disabled"}
              </span>
            </div>
          </div>
        );

      case "select":
        return (
          <div
            key={param.name}
            className="space-y-4 p-6 bg-gray-50 rounded-xl border"
          >
            <div>
              <label className="text-lg font-semibold text-gray-900">
                {param.label}
              </label>
              <p className="text-sm text-gray-600 mt-1">{param.description}</p>
            </div>
            <select
              value={currentValue}
              onChange={(e) =>
                updateEffectParameter(effectName, param.name, e.target.value)
              }
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg bg-white"
            >
              {param.options?.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
        );

      default:
        return null;
    }
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
          className="bg-white rounded-2xl shadow-2xl max-w-6xl w-full mx-4 max-h-[85vh] overflow-hidden flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-8 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-purple-50">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Edit Segment Effects
              </h2>
              <p className="text-lg text-gray-600">
                {formatTime(segment.startTime)} - {formatTime(segment.endTime)}
                <span className="text-gray-400 mx-2">•</span>
                Duration: {formatTime(segment.duration)}
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-3 hover:bg-gray-100 rounded-full transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          <div className="flex flex-1 min-h-0">
            {/* Left Panel - Effects List */}
            <div className="w-1/2 border-r border-gray-200 p-8 overflow-y-auto">
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-semibold text-gray-900">
                    Effects
                  </h3>
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={localSegment.enabled ?? true}
                      onChange={(e) => {
                        console.log(
                          "Enable segment changed:",
                          e.target.checked
                        );
                        updateSegment({ enabled: e.target.checked });
                      }}
                      className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    <span className="text-sm font-medium text-gray-700">
                      Enable Segment
                    </span>
                  </div>
                </div>

                <div className="space-y-4">
                  {Object.keys(AVAILABLE_EFFECTS).map((effectName) => {
                    const effectConfig = AVAILABLE_EFFECTS[effectName];
                    const isEnabled =
                      localSegment.effectCustomizations?.[effectName]
                        ?.enabled ?? false;

                    const Icon = effectConfig.icon;

                    return (
                      <div
                        key={effectName}
                        className={cn(
                          "border-2 rounded-xl p-6 cursor-pointer transition-all duration-200",
                          selectedEffect === effectName
                            ? "border-blue-500 bg-blue-50 shadow-lg"
                            : "border-gray-200 hover:border-gray-300 hover:shadow-md",
                          !isEnabled && "opacity-50 bg-gray-50"
                        )}
                        onClick={() => setSelectedEffect(effectName)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div
                              className={cn(
                                "p-3 rounded-lg",
                                selectedEffect === effectName
                                  ? "bg-blue-100"
                                  : "bg-gray-100"
                              )}
                            >
                              <Icon className="w-6 h-6 text-gray-600" />
                            </div>
                            <div>
                              <h4 className="font-semibold text-gray-900 text-lg">
                                {effectConfig.name}
                              </h4>
                              <p className="text-sm text-gray-500">
                                {effectName}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center space-x-3">
                            <input
                              type="checkbox"
                              checked={isEnabled}
                              onChange={(e) =>
                                updateEffectEnabled(
                                  effectName,
                                  e.target.checked
                                )
                              }
                              onClick={(e) => e.stopPropagation()}
                              className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                            />
                            {isEnabled ? (
                              <Eye className="w-5 h-5 text-green-500" />
                            ) : (
                              <EyeOff className="w-5 h-5 text-gray-400" />
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Right Panel - Parameters */}
            <div className="w-1/2 p-8 overflow-y-auto">
              {selectedEffect && AVAILABLE_EFFECTS[selectedEffect] ? (
                <div className="space-y-8">
                  <div className="flex items-center space-x-4">
                    <div className="p-3 bg-blue-100 rounded-lg">
                      {React.createElement(
                        AVAILABLE_EFFECTS[selectedEffect].icon,
                        { className: "w-8 h-8 text-blue-600" }
                      )}
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-gray-900">
                        {AVAILABLE_EFFECTS[selectedEffect].name}
                      </h3>
                      <p className="text-gray-600">Adjust effect parameters</p>
                    </div>
                  </div>

                  <div className="space-y-6">
                    {AVAILABLE_EFFECTS[selectedEffect].parameters.map((param) =>
                      renderParameterInput(param, selectedEffect)
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500">
                  <div className="text-center">
                    <div className="p-6 bg-gray-100 rounded-full w-20 h-20 mx-auto mb-6 flex items-center justify-center">
                      <Settings className="w-10 h-10 text-gray-300" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-400 mb-2">
                      Select an Effect
                    </h3>
                    <p className="text-gray-500">
                      Choose an effect from the left panel to adjust its
                      parameters
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end space-x-4 p-6 border-t border-gray-200 bg-gray-50">
            <Button
              variant="outline"
              onClick={handleCancel}
              className="px-6 py-2"
            >
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold shadow-lg"
            >
              💾 Save Changes
            </Button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}
