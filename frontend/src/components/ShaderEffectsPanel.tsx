"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Palette,
  Sparkles,
  Zap,
  Eye,
  EyeOff,
  Sliders,
  RotateCw,
  Layers,
  Target,
  Star,
  X,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface ShaderEffect {
  id: string;
  name: string;
  category:
    | "transition"
    | "filter"
    | "distortion"
    | "color"
    | "motion"
    | "atmospheric";
  intensity: number;
  enabled: boolean;
  parameters: Record<string, any>;
  description: string;
  preview?: string;
}

interface ShaderEffectsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  selectedEffects: ShaderEffect[];
  onEffectsChange: (effects: ShaderEffect[]) => void;
}

const AVAILABLE_SHADER_EFFECTS: ShaderEffect[] = [
  {
    id: "glitch",
    name: "Digital Glitch",
    category: "distortion",
    intensity: 0.5,
    enabled: false,
    parameters: { frequency: 0.1, intensity: 0.5, colorShift: 0.3 },
    description: "Digital corruption and glitch effects",
  },
  {
    id: "vintage",
    name: "Vintage Film",
    category: "color",
    intensity: 0.7,
    enabled: false,
    parameters: { saturation: 0.8, contrast: 1.2, grain: 0.3, vignette: 0.4 },
    description: "Classic film look with grain and color grading",
  },
  {
    id: "neon",
    name: "Neon Glow",
    category: "color",
    intensity: 0.6,
    enabled: false,
    parameters: {
      brightness: 1.3,
      saturation: 1.5,
      glow: 0.8,
      colorShift: 0.2,
    },
    description: "Vibrant neon colors with glow effects",
  },
  {
    id: "motion_blur",
    name: "Motion Blur",
    category: "motion",
    intensity: 0.4,
    enabled: false,
    parameters: { radius: 5, direction: "horizontal", strength: 0.6 },
    description: "Dynamic motion blur for action scenes",
  },
  {
    id: "pixelate",
    name: "Pixel Art",
    category: "distortion",
    intensity: 0.3,
    enabled: false,
    parameters: { blockSize: 8, colorDepth: 8, dither: true },
    description: "Retro pixel art style",
  },
  {
    id: "wave",
    name: "Wave Distortion",
    category: "distortion",
    intensity: 0.5,
    enabled: false,
    parameters: { amplitude: 10, frequency: 2, speed: 1.0 },
    description: "Liquid wave distortion effects",
  },
  {
    id: "zoom_blur",
    name: "Zoom Blur",
    category: "motion",
    intensity: 0.6,
    enabled: false,
    parameters: { center: [0.5, 0.5], strength: 0.8, falloff: 0.3 },
    description: "Radial zoom blur from center point",
  },
  {
    id: "chromatic",
    name: "Chromatic Aberration",
    category: "distortion",
    intensity: 0.4,
    enabled: false,
    parameters: { offset: 5, redShift: 0.3, blueShift: -0.3 },
    description: "Color channel separation effects",
  },
  {
    id: "fog",
    name: "Atmospheric Fog",
    category: "atmospheric",
    intensity: 0.5,
    enabled: false,
    parameters: { density: 0.3, color: [0.8, 0.8, 0.9], height: 0.5 },
    description: "Mysterious atmospheric fog",
  },
  {
    id: "light_leak",
    name: "Light Leak",
    category: "color",
    intensity: 0.4,
    enabled: false,
    parameters: { intensity: 0.6, color: [1.0, 0.8, 0.6], position: 0.7 },
    description: "Film camera light leak effect",
  },
  {
    id: "vhs",
    name: "VHS Effect",
    category: "distortion",
    intensity: 0.6,
    enabled: false,
    parameters: { tracking: 0.3, noise: 0.2, colorShift: 0.1, scanlines: true },
    description: "Retro VHS tape distortion",
  },
  {
    id: "dream",
    name: "Dream Sequence",
    category: "atmospheric",
    intensity: 0.7,
    enabled: false,
    parameters: { blur: 0.4, glow: 0.6, colorGrading: "warm", vignette: 0.3 },
    description: "Soft, dreamy atmospheric effect",
  },
];

const CATEGORY_ICONS = {
  transition: Sparkles,
  filter: Eye,
  distortion: Zap,
  color: Palette,
  motion: Target,
  atmospheric: Layers,
};

const CATEGORY_COLORS = {
  transition: "from-purple-500 to-pink-500",
  filter: "from-blue-500 to-cyan-500",
  distortion: "from-red-500 to-orange-500",
  color: "from-green-500 to-emerald-500",
  motion: "from-yellow-500 to-orange-500",
  atmospheric: "from-indigo-500 to-purple-500",
};

export function ShaderEffectsPanel({
  isOpen,
  onClose,
  selectedEffects,
  onEffectsChange,
}: ShaderEffectsPanelProps) {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set()
  );
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  const toggleEffect = (effectId: string) => {
    const updatedEffects = selectedEffects.map((effect) =>
      effect.id === effectId ? { ...effect, enabled: !effect.enabled } : effect
    );
    onEffectsChange(updatedEffects);
  };

  const updateEffectIntensity = (effectId: string, intensity: number) => {
    const updatedEffects = selectedEffects.map((effect) =>
      effect.id === effectId ? { ...effect, intensity } : effect
    );
    onEffectsChange(updatedEffects);
  };

  const updateEffectParameter = (
    effectId: string,
    paramName: string,
    value: any
  ) => {
    const updatedEffects = selectedEffects.map((effect) =>
      effect.id === effectId
        ? {
            ...effect,
            parameters: { ...effect.parameters, [paramName]: value },
          }
        : effect
    );
    onEffectsChange(updatedEffects);
  };

  const resetEffect = (effectId: string) => {
    const originalEffect = AVAILABLE_SHADER_EFFECTS.find(
      (e) => e.id === effectId
    );
    if (originalEffect) {
      const updatedEffects = selectedEffects.map((effect) =>
        effect.id === effectId
          ? { ...originalEffect, enabled: effect.enabled }
          : effect
      );
      onEffectsChange(updatedEffects);
    }
  };

  const filteredEffects = AVAILABLE_SHADER_EFFECTS.filter((effect) => {
    const matchesSearch =
      effect.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      effect.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory =
      !selectedCategory || effect.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const groupedEffects = filteredEffects.reduce((groups, effect) => {
    if (!groups[effect.category]) {
      groups[effect.category] = [];
    }
    groups[effect.category].push(effect);
    return groups;
  }, {} as Record<string, ShaderEffect[]>);

  const renderParameterControl = (
    effect: ShaderEffect,
    paramName: string,
    paramValue: any
  ) => {
    if (typeof paramValue === "number") {
      return (
        <div key={paramName} className="space-y-1">
          <label className="text-xs text-gray-600 capitalize">
            {paramName.replace(/([A-Z])/g, " $1").trim()}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={paramValue}
            onChange={(e) =>
              updateEffectParameter(
                effect.id,
                paramName,
                parseFloat(e.target.value)
              )
            }
            className="w-full"
          />
          <span className="text-xs text-gray-500">
            {Math.round(paramValue * 100)}%
          </span>
        </div>
      );
    } else if (typeof paramValue === "boolean") {
      return (
        <div key={paramName} className="flex items-center justify-between">
          <label className="text-xs text-gray-600 capitalize">
            {paramName.replace(/([A-Z])/g, " $1").trim()}
          </label>
          <input
            type="checkbox"
            checked={paramValue}
            onChange={(e) =>
              updateEffectParameter(effect.id, paramName, e.target.checked)
            }
            className="rounded"
          />
        </div>
      );
    }
    return null;
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, x: 300 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 300 }}
          className="fixed right-0 top-0 h-full w-96 bg-white shadow-xl border-l z-50"
        >
          <div className="h-full flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b">
              <div className="flex items-center gap-2">
                <Palette className="w-5 h-5 text-purple-500" />
                <h3 className="text-lg font-semibold">Shader Effects</h3>
              </div>
              <Button variant="ghost" size="sm" onClick={onClose}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Search and Filter */}
            <div className="p-4 border-b space-y-3">
              <input
                type="text"
                placeholder="Search effects..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-2 border rounded-md text-sm"
              />

              <div className="flex flex-wrap gap-2">
                {Object.keys(CATEGORY_ICONS).map((category) => {
                  const Icon =
                    CATEGORY_ICONS[category as keyof typeof CATEGORY_ICONS];
                  const isSelected = selectedCategory === category;
                  return (
                    <Button
                      key={category}
                      size="sm"
                      variant={isSelected ? "default" : "outline"}
                      onClick={() =>
                        setSelectedCategory(isSelected ? null : category)
                      }
                      className="text-xs"
                    >
                      <Icon className="w-3 h-3 mr-1" />
                      {category}
                    </Button>
                  );
                })}
              </div>
            </div>

            {/* Effects List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {Object.entries(groupedEffects).map(([category, effects]) => {
                const Icon =
                  CATEGORY_ICONS[category as keyof typeof CATEGORY_ICONS];
                const isExpanded = expandedCategories.has(category);
                const enabledCount = effects.filter((e) => e.enabled).length;

                return (
                  <Card key={category} className="overflow-hidden">
                    <CardHeader
                      className="cursor-pointer hover:bg-gray-50 transition-colors"
                      onClick={() => toggleCategory(category)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-8 h-8 rounded-full bg-gradient-to-r ${
                              CATEGORY_COLORS[
                                category as keyof typeof CATEGORY_COLORS
                              ]
                            } flex items-center justify-center`}
                          >
                            <Icon className="w-4 h-4 text-white" />
                          </div>
                          <div>
                            <CardTitle className="text-sm capitalize">
                              {category}
                            </CardTitle>
                            <CardDescription className="text-xs">
                              {enabledCount} of {effects.length} enabled
                            </CardDescription>
                          </div>
                        </div>
                        {isExpanded ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        )}
                      </div>
                    </CardHeader>

                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                        >
                          <CardContent className="pt-0 space-y-3">
                            {effects.map((effect) => {
                              const isEnabled =
                                selectedEffects.find((e) => e.id === effect.id)
                                  ?.enabled || false;
                              const currentEffect =
                                selectedEffects.find(
                                  (e) => e.id === effect.id
                                ) || effect;

                              return (
                                <div
                                  key={effect.id}
                                  className="border rounded-lg p-3 space-y-3"
                                >
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                      <input
                                        type="checkbox"
                                        checked={isEnabled}
                                        onChange={() => toggleEffect(effect.id)}
                                        className="rounded"
                                      />
                                      <span className="text-sm font-medium">
                                        {effect.name}
                                      </span>
                                    </div>
                                    {isEnabled && (
                                      <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => resetEffect(effect.id)}
                                        className="text-xs"
                                      >
                                        <RotateCw className="w-3 h-3" />
                                      </Button>
                                    )}
                                  </div>

                                  {isEnabled && (
                                    <div className="space-y-3">
                                      <div>
                                        <label className="text-xs text-gray-600">
                                          Intensity
                                        </label>
                                        <div className="flex items-center gap-2">
                                          <input
                                            type="range"
                                            min="0"
                                            max="1"
                                            step="0.1"
                                            value={currentEffect.intensity}
                                            onChange={(e) =>
                                              updateEffectIntensity(
                                                effect.id,
                                                parseFloat(e.target.value)
                                              )
                                            }
                                            className="flex-1"
                                          />
                                          <span className="text-xs text-gray-500 w-8">
                                            {Math.round(
                                              currentEffect.intensity * 100
                                            )}
                                            %
                                          </span>
                                        </div>
                                      </div>

                                      <div className="space-y-2">
                                        <label className="text-xs text-gray-600">
                                          Parameters
                                        </label>
                                        {Object.entries(
                                          currentEffect.parameters
                                        ).map(([paramName, paramValue]) =>
                                          renderParameterControl(
                                            currentEffect,
                                            paramName,
                                            paramValue
                                          )
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  <p className="text-xs text-gray-500">
                                    {effect.description}
                                  </p>
                                </div>
                              );
                            })}
                          </CardContent>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </Card>
                );
              })}
            </div>

            {/* Footer */}
            <div className="p-4 border-t bg-gray-50">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">
                  {selectedEffects.filter((e) => e.enabled).length} effects
                  active
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    const resetEffects = selectedEffects.map((effect) => ({
                      ...effect,
                      enabled: false,
                    }));
                    onEffectsChange(resetEffects);
                  }}
                >
                  Clear All
                </Button>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
