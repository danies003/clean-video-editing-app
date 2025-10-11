// Shared types for the enhanced video editor

export interface TimelineSegment {
  id: string;
  startTime: number;
  endTime: number;
  duration: number;
  transitionIn?: string;
  transitionOut?: string;
  effects?: string[];
  thumbnail?: string;
}

export interface StyleTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<any>;
  category: "social" | "cinematic" | "music" | "custom";
  preview: string;
  settings: {
    energy_level: "low" | "medium" | "high";
    transition_style: "smooth" | "dynamic" | "aggressive";
    pacing: "slow" | "medium" | "fast";
    edit_scale: number;
    target_duration?: number;
  };
}

export interface ExportSettings {
  format: string;
  quality: string;
  resolution: string;
  bitrate: number;
  fps: number;
  audio: boolean;
  customSettings: {
    targetDuration?: number;
    aspectRatio?: string;
    colorGrading?: string;
    audioNormalization?: boolean;
  };
}

export type EditingStep =
  | "analysis"
  | "style_selection"
  | "timeline_editing"
  | "preview"
  | "export"
  | "completed"
  | "error";

export interface EditingState {
  step: EditingStep;
  analysisJob?: any;
  editingJob?: any;
  error?: string;
  resultUrl?: string;
  timeline?: any;
  analysisResult?: any;
}
