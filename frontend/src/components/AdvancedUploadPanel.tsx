"use client";

import React, { useState, useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Switch } from "./ui/switch";
import { Button } from "./ui/button";
import { Upload, Video, X } from "lucide-react";

// Define the parameter types based on the backend schemas
type EditStyle = "tiktok" | "youtube" | "cinematic";
type TemplateType =
  | "beat_match"
  | "cinematic"
  | "fast_paced"
  | "slow_motion"
  | "transition_heavy"
  | "minimal";
type QualityPreset = "low" | "medium" | "high" | "ultra";

interface AdvancedUploadPanelProps {
  onUpload: (params: UploadParams) => void;
  onMultiUpload: (files: File[]) => void;
  isUploading: boolean;
  uploadProgress: string;
  analysisProgress?: number;
}

interface UploadParams {
  file: File;
  editStyle: EditStyle;
  templateType: TemplateType;
  qualityPreset: QualityPreset;
  skipPreview: boolean;
}

interface MultiUploadParams {
  files: File[];
  editStyle: EditStyle;
  templateType: TemplateType;
  qualityPreset: QualityPreset;
  crossVideoSettings: {
    enableCrossAnalysis: boolean;
    similarityThreshold: number;
    chunkingStrategy: "scene" | "action" | "audio" | "content";
  };
}

interface VideoFile {
  file: File;
  id: string;
  size: string;
  status: "pending" | "uploading" | "uploaded" | "error";
}

const AdvancedUploadPanel: React.FC<AdvancedUploadPanelProps> = ({
  onUpload,
  onMultiUpload,
  isUploading,
  uploadProgress,
  analysisProgress,
}) => {
  const [editStyle, setEditStyle] = useState<EditStyle>("tiktok");
  const [templateType, setTemplateType] = useState<TemplateType>("beat_match");
  const [qualityPreset, setQualityPreset] = useState<QualityPreset>("high");
  const [skipPreview, setSkipPreview] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<VideoFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);

  // Multi-video settings (automatically enabled when multiple files)
  const [enableCrossAnalysis, setEnableCrossAnalysis] = useState(true);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [chunkingStrategy, setChunkingStrategy] = useState<
    "scene" | "action" | "audio" | "content"
  >("scene");

  // Auto-detect if multi-video mode should be enabled
  const isMultiVideoMode = selectedFiles.length > 1;

  const handleFileSelect = (files: FileList | File[]) => {
    const newFiles: VideoFile[] = Array.from(files)
      .filter((file) => file.type.startsWith("video/"))
      .map((file) => ({
        file,
        id: Math.random().toString(36).substr(2, 9),
        size: formatFileSize(file.size),
        status: "pending" as const,
      }));

    // Always add to existing files (no replacement)
    setSelectedFiles((prev) => [...prev, ...newFiles]);
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files);
    }
  }, []);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      handleFileSelect(Array.from(files));
    }
  };

  const handleUpload = async () => {
    console.log("🔧 [DEBUG] ===== UPLOAD HANDLER START =====");
    console.log(
      "🔧 [DEBUG] handleUpload called with files:",
      selectedFiles.length
    );
    console.log("🔧 [DEBUG] isMultiVideoMode:", isMultiVideoMode);
    console.log("🔧 [DEBUG] Current window location:", window.location.href);
    console.log(
      "🔧 [DEBUG] Current window pathname:",
      window.location.pathname
    );

    if (selectedFiles.length === 0) {
      console.log("🔧 [DEBUG] No files selected, showing alert");
      alert("Please select at least one video file");
      return;
    }

    try {
      if (isMultiVideoMode) {
        console.log("🔧 [DEBUG] ===== MULTI-VIDEO UPLOAD PATH =====");
        console.log(
          "🔧 [DEBUG] Calling onMultiUpload with files:",
          selectedFiles.map((vf) => vf.file.name)
        );
        console.log("🔧 [DEBUG] About to call onMultiUpload...");

        // Multi-video upload
        onMultiUpload(selectedFiles.map((vf) => vf.file));

        console.log("🔧 [DEBUG] onMultiUpload call completed");
      } else {
        console.log("🔧 [DEBUG] ===== SINGLE VIDEO UPLOAD PATH =====");
        console.log("🔧 [DEBUG] Calling onUpload for single video");
        console.log("🔧 [DEBUG] File details:", {
          name: selectedFiles[0].file.name,
          size: selectedFiles[0].file.size,
          type: selectedFiles[0].file.type,
        });

        // Single video upload
        const params: UploadParams = {
          file: selectedFiles[0].file,
          editStyle,
          templateType,
          qualityPreset,
          skipPreview,
        };

        console.log("🔧 [DEBUG] Upload params:", params);
        console.log("🔧 [DEBUG] About to call onUpload...");

        onUpload(params);

        console.log("🔧 [DEBUG] onUpload call completed");
      }
    } catch (error) {
      console.error("🔧 [DEBUG] Error in handleUpload:", error);
      console.error(
        "🔧 [DEBUG] Error stack:",
        error instanceof Error ? error.stack : "No stack trace"
      );
    }

    console.log("🔧 [DEBUG] ===== UPLOAD HANDLER END =====");
  };

  const removeFile = (id: string) => {
    setSelectedFiles((prev) => prev.filter((file) => file.id !== id));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Advanced Video Upload</CardTitle>
        <CardDescription>
          Upload your video and configure editing parameters
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* File Upload Area */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Video File</label>

          {!selectedFiles.length ? (
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragOver
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-300 hover:border-gray-400"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-900 mb-2">
                Drop your videos here
              </p>
              <p className="text-gray-500 mb-4">or click to browse files</p>
              <input
                type="file"
                multiple={true}
                accept="video/*"
                onChange={handleFileInputChange}
                className="hidden"
                id="video-upload"
              />
              <label
                htmlFor="video-upload"
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 cursor-pointer"
              >
                Choose Files
              </label>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-gray-900">
                  {selectedFiles.length === 1
                    ? "Selected Video"
                    : `Selected Videos (${selectedFiles.length})`}
                </h3>
                {selectedFiles.length > 1 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedFiles([])}
                    className="text-red-500 hover:text-red-700"
                  >
                    Clear All
                  </Button>
                )}
              </div>
              <div className="space-y-2">
                {selectedFiles.map((videoFile) => (
                  <div
                    key={videoFile.id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <Video className="h-5 w-5 text-gray-500" />
                      <div>
                        <p className="font-medium text-gray-900">
                          {videoFile.file.name}
                        </p>
                        <p className="text-sm text-gray-500">
                          {videoFile.size}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(videoFile.id)}
                      className="text-red-500 hover:text-red-700"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Edit Style Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Edit Style</label>
          <Select
            value={editStyle}
            onValueChange={(value: EditStyle) => setEditStyle(value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select edit style" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="tiktok">TikTok</SelectItem>
              <SelectItem value="youtube">YouTube</SelectItem>
              <SelectItem value="cinematic">Cinematic</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {editStyle === "tiktok" &&
              "Optimized for short-form vertical content"}
            {editStyle === "youtube" &&
              "Perfect for longer-form horizontal videos"}
            {editStyle === "cinematic" &&
              "Professional cinematic editing style"}
          </p>
        </div>

        {/* Template Type Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Template Type</label>
          <Select
            value={templateType}
            onValueChange={(value: TemplateType) => setTemplateType(value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select template type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="beat_match">Beat Match</SelectItem>
              <SelectItem value="cinematic">Cinematic</SelectItem>
              <SelectItem value="fast_paced">Fast Paced</SelectItem>
              <SelectItem value="slow_motion">Slow Motion</SelectItem>
              <SelectItem value="transition_heavy">Transition Heavy</SelectItem>
              <SelectItem value="minimal">Minimal</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {templateType === "beat_match" &&
              "Synchronizes cuts with audio beats"}
            {templateType === "cinematic" && "Professional movie-style editing"}
            {templateType === "fast_paced" &&
              "Quick cuts and dynamic transitions"}
            {templateType === "slow_motion" && "Emphasizes slow motion effects"}
            {templateType === "transition_heavy" &&
              "Focuses on creative transitions"}
            {templateType === "minimal" && "Clean, simple editing approach"}
          </p>
        </div>

        {/* Quality Preset Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Quality Preset</label>
          <Select
            value={qualityPreset}
            onValueChange={(value: QualityPreset) => setQualityPreset(value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select quality preset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="low">Low</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="ultra">Ultra</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {qualityPreset === "low" && "Fast processing, lower quality"}
            {qualityPreset === "medium" && "Balanced quality and speed"}
            {qualityPreset === "high" &&
              "High quality, moderate processing time"}
            {qualityPreset === "ultra" &&
              "Maximum quality, longer processing time"}
          </p>
        </div>

        {/* Skip Preview Option */}
        <div className="flex items-center space-x-2">
          <Switch
            id="skip-preview"
            checked={skipPreview}
            onCheckedChange={setSkipPreview}
          />
          <label htmlFor="skip-preview" className="text-sm font-medium">
            Skip Preview
          </label>
        </div>
        <p className="text-xs text-muted-foreground">
          {skipPreview
            ? "Video will be processed directly without showing the preview page"
            : "You will see a preview page with LLM suggestions before processing"}
        </p>

        {/* Multi-video settings */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Multi-Video Settings</label>
          <div className="flex items-center space-x-2">
            <Switch
              id="enable-multi-video"
              checked={isMultiVideoMode}
              onCheckedChange={() => {}}
              disabled={true}
            />
            <label htmlFor="enable-multi-video" className="text-sm font-medium">
              {isMultiVideoMode
                ? `Multi-Video Mode (${selectedFiles.length} videos detected)`
                : "Single Video Mode"}
            </label>
          </div>
          {isMultiVideoMode && (
            <>
              <div className="flex items-center space-x-2">
                <Switch
                  id="enable-cross-analysis"
                  checked={enableCrossAnalysis}
                  onCheckedChange={setEnableCrossAnalysis}
                />
                <label
                  htmlFor="enable-cross-analysis"
                  className="text-sm font-medium"
                >
                  Enable Cross-Video Analysis
                </label>
              </div>
              <div className="flex items-center space-x-2">
                <label
                  htmlFor="similarity-threshold"
                  className="text-sm font-medium"
                >
                  Similarity Threshold:
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={similarityThreshold}
                  onChange={(e) =>
                    setSimilarityThreshold(parseFloat(e.target.value))
                  }
                  className="w-24"
                />
                <span>{similarityThreshold.toFixed(2)}</span>
              </div>
              <div className="flex items-center space-x-2">
                <label
                  htmlFor="chunking-strategy"
                  className="text-sm font-medium"
                >
                  Chunking Strategy:
                </label>
                <Select
                  value={chunkingStrategy}
                  onValueChange={(value) =>
                    setChunkingStrategy(
                      value as "scene" | "action" | "audio" | "content"
                    )
                  }
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Select chunking strategy" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="scene">Scene-based</SelectItem>
                    <SelectItem value="action">Action-based</SelectItem>
                    <SelectItem value="audio">Audio-based</SelectItem>
                    <SelectItem value="content">Content-based</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </>
          )}
        </div>

        {/* Progress Display */}
        {isUploading && (uploadProgress || analysisProgress !== undefined) && (
          <div className="space-y-3 p-4 bg-blue-50 rounded-lg border border-blue-200">
            {uploadProgress && (
              <div className="text-sm text-blue-800">
                <div className="font-medium mb-1">Status:</div>
                <div>{uploadProgress}</div>
              </div>
            )}
            {analysisProgress !== undefined && (
              <div className="space-y-2">
                <div className="text-sm text-blue-800 font-medium">
                  AI Processing Progress: {Math.round(analysisProgress)}%
                </div>
                <div className="w-full bg-blue-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${analysisProgress}%` }}
                  ></div>
                </div>
                <div className="text-xs text-blue-600">
                  {analysisProgress < 25 && "Analyzing videos..."}
                  {analysisProgress >= 25 &&
                    analysisProgress < 50 &&
                    "Cross-video analysis..."}
                  {analysisProgress >= 50 &&
                    analysisProgress < 75 &&
                    "Preparing editing..."}
                  {analysisProgress >= 75 &&
                    analysisProgress < 100 &&
                    "Applying effects and transitions..."}
                  {analysisProgress >= 100 && "Processing completed!"}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Upload Button */}
        <Button
          type="button"
          onClick={handleUpload}
          disabled={isUploading || selectedFiles.length === 0}
          className="w-full"
        >
          {isUploading ? "Processing..." : "Upload Video"}
        </Button>
      </CardContent>
    </Card>
  );
};

export default AdvancedUploadPanel;
