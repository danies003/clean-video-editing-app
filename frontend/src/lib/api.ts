// API client for FastAPI backend with proper backend discovery and health checks

// Environment configuration
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

console.log("🔧 Using API URL:", API_URL);

// Types matching the FastAPI backend
export interface VideoUploadResponse {
  success: boolean;
  message: string;
  timestamp: string;
  video_id: string;
  upload_url: string;
  expires_at: string;
  metadata?: {
    filename: string;
    size: number;
    content_type?: string;
    duration?: number;
  };
}

export interface VideoAnalysisResult {
  video_id: string;
  analysis: {
    beats: Array<{
      timestamp: number;
      confidence: number;
      energy: number;
    }>;
    motion_spikes: Array<{
      timestamp: number;
      intensity: number;
      type: string;
    }>;
    scene_changes: Array<{
      timestamp: number;
      confidence: number;
      visual_similarity: number;
    }>;
  };
  metadata: {
    duration: number;
    fps: number;
    resolution: [number, number];
  };
}

export interface ProcessingJob {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  message?: string;
  output_url?: string;
  analysis_result?: unknown;
  timeline?: unknown;
  error?: string;
  created_at: string;
  updated_at: string;
  metadata?: {
    analysis_mode?: string;
    analysis_error?: string;
    [key: string]: unknown;
  };
  result?: {
    output_url?: string;
    analysis_result?: unknown;
    timeline?: unknown;
  };
}

export interface JobStatusResponse {
  job?: ProcessingJob;
}

export interface AdvancedEditRequest {
  video_id: string;
  edit_type?: string;
  edit_scale?: number;
  style_preferences?: {
    style: string;
    energy_level: string;
    transition_style: string;
    pacing: string;
  };
  target_duration?: number;
  parameters?: Record<string, any>;
  segments?: Array<{
    start_time: number;
    end_time: number;
    effects?: Record<string, any>;
  }>;
}

export interface CompletedVideoResponse {
  video_id: string;
  status: "completed" | "failed" | "processing" | "pending";
  output_url?: string;
  analysis?: unknown;
  timeline?: unknown;
  metadata?: {
    rq_job_id?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface EditVideoRequest {
  video_id: string;
  template_type:
    | "fast_beat_match"
    | "cinematic_cut"
    | "smooth_flow"
    | "dynamic_energy";
  quality_preset: "low" | "medium" | "high" | "ultra";
  custom_settings?: Record<string, unknown>;
}

export interface AnalyzeVideoRequest {
  template_type:
    | "beat_match"
    | "cinematic"
    | "fast_paced"
    | "slow_motion"
    | "transition_heavy"
    | "minimal";
  analysis_options: {
    edit_style: "tiktok" | "youtube" | "cinematic";
    quality_preset: "low" | "medium" | "high" | "ultra";
  };
  video_url: string;
}

export interface VideoUploadRequest {
  filename: string;
  file_size: number;
  content_type: string;
  template_type?:
    | "beat_match"
    | "cinematic"
    | "fast_paced"
    | "slow_motion"
    | "transition_heavy"
    | "minimal";
  quality_preset: "low" | "medium" | "high" | "ultra";
  custom_settings?: Record<string, unknown>;
  video_url?: string;
}

// Multi-Video Project types
export interface MultiVideoProject {
  project_id: string;
  name: string;
  video_ids: string[];
  status: "pending" | "analyzing" | "editing" | "completed" | "failed";
  analysis_jobs: string[];
  cross_analysis_job?: string;
  editing_job?: string;
  output_video_id?: string;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

export interface MultiVideoProjectStatus {
  project_id: string;
  status: "pending" | "analyzing" | "editing" | "completed" | "failed";
  video_ids: string[]; // List of video IDs in the project
  analysis_completed: number;
  cross_analysis_completed: boolean;
  editing_completed: boolean;
  progress: number;
  output_video_url?: string;
  error?: string;
  metadata?: Record<string, any>;
}

export interface CrossVideoSettings {
  enableCrossAnalysis: boolean;
  similarityThreshold: number;
  chunkingStrategy: "scene" | "action" | "audio" | "content";
}

export interface MultiVideoAnalysisSettings {
  similarityThreshold: number;
  chunkingStrategy: "scene" | "action" | "audio" | "content";
  crossAnalysisSettings: {
    enableCrossAnalysis: boolean;
  };
}

export interface MultiVideoEditSettings {
  editScale: number;
  stylePreferences: {
    style: "tiktok" | "youtube" | "cinematic";
    energyLevel: "low" | "medium" | "high";
    transitionStyle: "smooth" | "dynamic" | "aggressive";
    pacing: "slow" | "medium" | "fast";
  };
  crossVideoEffects: {
    enableTransitions: boolean;
    enableEffects: boolean;
  };
  targetDuration?: number;
}

// Main API Client Class
class APIClient {
  private baseURL: string;
  private backendURLs: string[] = [
    "http://localhost:8000",
    "http://localhost:8001",
  ];

  constructor() {
    this.baseURL = API_URL;
  }

  // Backend discovery and health check methods
  async getCurrentBackendURL(): Promise<string> {
    // Try the configured API_URL first
    try {
      const response = await fetch(`${this.baseURL}/health/simple`);
      if (response.ok) {
        console.log("✅ Primary backend URL working:", this.baseURL);
        return this.baseURL;
      }
    } catch (error) {
      console.log("❌ Primary backend URL failed:", this.baseURL);
    }

    // Try alternative URLs
    for (const url of this.backendURLs) {
      if (url === this.baseURL) continue; // Skip the primary URL

      try {
        console.log(`🔍 Trying backend URL: ${url}`);
        const response = await fetch(`${url}/health/simple`);
        if (response.ok) {
          console.log(`✅ Found working backend: ${url}`);
          this.baseURL = url; // Update the base URL
          return url;
        }
      } catch (error) {
        console.log(`❌ Backend URL failed: ${url}`);
      }
    }

    throw new Error("No working backend found");
  }

  async healthCheck(): Promise<void> {
    const url = await this.getCurrentBackendURL();
    const response = await fetch(`${url}/health/simple`);

    if (!response.ok) {
      throw new Error(
        `Health check failed: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();
    if (data.status !== "healthy") {
      throw new Error(`Backend unhealthy: ${data.message || "Unknown error"}`);
    }
  }

  // Video upload methods
  async uploadVideo(file: File): Promise<VideoUploadResponse> {
    const url = await this.getCurrentBackendURL();

    // Create upload request
    const uploadRequest: VideoUploadRequest = {
      filename: file.name,
      file_size: file.size,
      content_type: file.type,
      template_type: "beat_match",
      quality_preset: "high",
    };

    const response = await fetch(`${url}/api/v1/videos/upload`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(uploadRequest),
    });

    if (!response.ok) {
      throw new Error(
        `Upload failed: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }

  async analyzeVideo(
    videoId: string,
    request: AnalyzeVideoRequest
  ): Promise<JobStatusResponse> {
    const url = await this.getCurrentBackendURL();

    const response = await fetch(`${url}/api/v1/videos/${videoId}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(
        `Analysis failed: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }

  async getVideoStatus(videoId: string): Promise<JobStatusResponse> {
    const url = await this.getCurrentBackendURL();

    const response = await fetch(`${url}/api/v1/videos/${videoId}/status`);

    if (!response.ok) {
      throw new Error(
        `Status check failed: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }

  async getJobStatus(videoId: string): Promise<JobStatusResponse> {
    return this.getVideoStatus(videoId);
  }

  async advancedEdit(request: AdvancedEditRequest): Promise<JobStatusResponse> {
    const url = await this.getCurrentBackendURL();

    const response = await fetch(
      `${url}/api/v1/videos/${request.video_id}/advanced-edit`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      }
    );

    if (!response.ok) {
      throw new Error(
        `Advanced edit failed: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }

  async clearCache(): Promise<void> {
    // Reset the base URL to force rediscovery
    this.baseURL = API_URL;
    console.log("🔄 Cache cleared, will rediscover backend on next request");
  }

  async forceLocalhost(): Promise<void> {
    // Force localhost backend
    this.baseURL = "http://localhost:8000";
    console.log("🏠 Forced localhost backend");
  }

  async getVideoDownloadUrl(videoId: string): Promise<string> {
    const url = await this.getCurrentBackendURL();
    return `${url}/api/v1/videos/${videoId}/download`;
  }

  async llmEdit(request: AdvancedEditRequest): Promise<JobStatusResponse> {
    const url = await this.getCurrentBackendURL();

    const response = await fetch(
      `${url}/api/v1/videos/${request.video_id}/llm-edit`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      }
    );

    if (!response.ok) {
      throw new Error(
        `LLM edit failed: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }

  async editVideo(
    videoId: string,
    request: EditVideoRequest
  ): Promise<JobStatusResponse> {
    const url = await this.getCurrentBackendURL();

    const response = await fetch(`${url}/api/v1/videos/${videoId}/edit`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Edit failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  // Multi-video project methods - following the established process
  async createProject(
    files: File[],
    projectName: string,
    settings: {
      enableCrossAnalysis?: boolean;
      similarityThreshold?: number;
      chunkingStrategy?: string;
    } = {}
  ): Promise<{ success: boolean; project_id: string; message: string }> {
    const url = await this.getCurrentBackendURL();

    // Convert old settings format to backend format
    const crossVideoSettings = {
      enableCrossAnalysis: settings.enableCrossAnalysis || false,
      similarityThreshold: settings.similarityThreshold || 0.7,
      chunkingStrategy: settings.chunkingStrategy || "scene",
    };

    const formData = new FormData();
    formData.append("project_name", projectName);
    formData.append("cross_video_settings", JSON.stringify(crossVideoSettings));

    files.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await fetch(`${url}/api/v1/multi-video/projects`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          `Failed to create project: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();
      return {
        success: true,
        project_id: result.project_id,
        message: "Project created successfully",
      };
    } catch (error) {
      return {
        success: false,
        project_id: "",
        message:
          error instanceof Error ? error.message : "Failed to create project",
      };
    }
  }

  async getProjectStatus(projectId: string): Promise<MultiVideoProjectStatus> {
    const url = await this.getCurrentBackendURL();

    const response = await fetch(
      `${url}/api/v1/multi-video/projects/${projectId}/status`
    );

    if (!response.ok) {
      throw new Error(
        `Failed to get project status: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }

  async getProjectTimeline(projectId: string): Promise<any> {
    const url = await this.getCurrentBackendURL();

    const response = await fetch(
      `${url}/api/v1/multi-video/projects/${projectId}/timeline`
    );

    if (!response.ok) {
      throw new Error(
        `Failed to get project timeline: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }

  // Legacy methods for backward compatibility
  async createMultiVideoProject(
    name: string,
    files: File[],
    settings: MultiVideoAnalysisSettings
  ): Promise<MultiVideoProject> {
    // Convert to the established process format
    const result = await this.createProject(files, name, {
      enableCrossAnalysis: settings.crossAnalysisSettings.enableCrossAnalysis,
      similarityThreshold: settings.similarityThreshold,
      chunkingStrategy: settings.chunkingStrategy,
    });

    if (!result.success) {
      throw new Error(result.message);
    }

    // Return a mock MultiVideoProject since the backend doesn't return this format
    return {
      project_id: result.project_id,
      name: name,
      video_ids: [],
      status: "pending",
      analysis_jobs: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
  }

  // Backward compatibility method - alias for getProjectStatus
  async getMultiVideoProjectStatus(projectId: string): Promise<any> {
    return this.getProjectStatus(projectId);
  }

  // Backward compatibility method - alias for getProjectTimeline
  async getMultiVideoProjectTimeline(projectId: string): Promise<any> {
    return this.getProjectTimeline(projectId);
  }

  // Download video with custom effects applied
  async downloadVideoWithCustomEffects(
    videoId: string,
    segments: any[],
    qualityPreset: "low" | "medium" | "high" = "high"
  ): Promise<{
    success: boolean;
    download_url?: string;
    render_job_id?: string;
    filename?: string;
    message: string;
  }> {
    try {
      const url = await this.getCurrentBackendURL();

      const response = await fetch(
        `${url}/api/v1/videos/${videoId}/download-with-custom-effects`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            segments: segments,
            quality_preset: qualityPreset,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Failed to render video with custom effects: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();
      return {
        success: true,
        download_url: result.download_url,
        render_job_id: result.render_job_id,
        filename: result.filename,
        message: result.message,
      };
    } catch (error) {
      return {
        success: false,
        message:
          error instanceof Error
            ? error.message
            : "Failed to render video with custom effects",
      };
    }
  }

  // Download multi-video project with custom effects applied
  async downloadMultiVideoWithCustomEffects(
    projectId: string,
    segments: any[],
    qualityPreset: "low" | "medium" | "high" = "high"
  ): Promise<{
    success: boolean;
    download_url?: string;
    render_job_id?: string;
    filename?: string;
    message: string;
  }> {
    try {
      const url = await this.getCurrentBackendURL();

      const response = await fetch(
        `${url}/api/v1/multi-video/projects/${projectId}/download-with-custom-effects`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            segments: segments,
            quality_preset: qualityPreset,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Failed to render multi-video with custom effects: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();
      return {
        success: true,
        download_url: result.download_url,
        render_job_id: result.render_job_id,
        filename: result.filename,
        message: result.message,
      };
    } catch (error) {
      return {
        success: false,
        message:
          error instanceof Error
            ? error.message
            : "Failed to render multi-video with custom effects",
      };
    }
  }
}

// Export singleton instance
export const apiClient = new APIClient();

// Export multiVideoApi instance for backward compatibility
export const multiVideoApi = new APIClient();
