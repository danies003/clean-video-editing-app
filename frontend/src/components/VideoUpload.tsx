"use client";

import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileVideo, X, CheckCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { apiClient, VideoUploadResponse } from "@/lib/api";

interface VideoUploadProps {
  onUploadComplete: (result: VideoUploadResponse) => void;
  className?: string;
}

export function VideoUpload({ onUploadComplete, className }: VideoUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      setError(null);
      setUploadedFile(file);
      setUploading(true);
      setUploadProgress(0);

      // Declare progressInterval outside try-catch for proper cleanup
      let progressInterval: NodeJS.Timeout | undefined;

      try {
        // Simulate upload progress up to 80%, then slow down
        progressInterval = setInterval(() => {
          setUploadProgress((prev) => {
            // Stop at 95% to allow real upload to complete
            if (prev >= 95) {
              return 95;
            }
            if (prev >= 80) {
              // Slow down progress after 80% to allow real upload to complete
              return prev + Math.random() * 2;
            }
            return prev + Math.random() * 10;
          });
        }, 200);

        const result = await apiClient.uploadVideo(file);

        // Clear interval and set to 100% immediately
        clearInterval(progressInterval);
        setUploadProgress(100);

        // Small delay to show 100% completion
        setTimeout(() => {
          onUploadComplete(result);
          setUploading(false);
        }, 500);
      } catch (error) {
        if (progressInterval) {
          clearInterval(progressInterval);
        }
        setError(error instanceof Error ? error.message : "Upload failed");
        setUploading(false);
        setUploadProgress(0);
      }
    },
    [onUploadComplete]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/*": [
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".wmv",
        ".flv",
        ".m4v",
        ".3gp",
        ".ogv",
        ".ts",
        ".mts",
        ".m2ts",
        ".vob",
        ".asf",
        ".rm",
        ".rmvb",
        ".f4v",
      ],
    },
    maxFiles: 1,
    disabled: uploading,
  });

  const resetUpload = () => {
    setUploadedFile(null);
    setError(null);
    setUploadProgress(0);
  };

  return (
    <div className={cn("w-full max-w-2xl mx-auto", className)}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative"
      >
        <div
          {...getRootProps()}
          className={cn(
            "relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300",
            "bg-gradient-to-br from-gray-50 to-gray-100 hover:from-blue-50 hover:to-purple-50",
            "backdrop-blur-sm border-gray-300 hover:border-blue-400",
            isDragActive && "border-blue-500 bg-blue-50 scale-[1.02]",
            uploading && "pointer-events-none opacity-75"
          )}
        >
          <input {...getInputProps()} />

          <AnimatePresence mode="wait">
            {uploading ? (
              <motion.div
                key="uploading"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="space-y-4"
              >
                <div className="relative w-16 h-16 mx-auto">
                  <div className="absolute inset-0 rounded-full border-4 border-gray-200"></div>
                  <motion.div
                    className="absolute inset-0 rounded-full border-4 border-blue-500 border-t-transparent"
                    animate={{ rotate: 360 }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  ></motion.div>
                </div>
                <div className="space-y-2">
                  <p className="text-lg font-medium text-gray-700">
                    {uploadProgress < 50 ? "Uploading" : "Converting"}{" "}
                    {uploadedFile?.name}...
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <motion.div
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${uploadProgress}%` }}
                      transition={{ duration: 0.3 }}
                    ></motion.div>
                  </div>
                  <p className="text-sm text-gray-500">
                    {uploadProgress < 50
                      ? `${Math.round(uploadProgress)}% uploaded`
                      : `${Math.round(uploadProgress)}% converted`}
                  </p>
                </div>
              </motion.div>
            ) : uploadProgress === 100 ? (
              <motion.div
                key="success"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-4"
              >
                <CheckCircle className="w-16 h-16 mx-auto text-green-500" />
                <p className="text-lg font-medium text-green-700">
                  Upload successful!
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="upload"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-4"
              >
                <div className="relative">
                  <motion.div
                    animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
                    className="w-16 h-16 mx-auto text-gray-400"
                  >
                    {isDragActive ? (
                      <Upload className="w-full h-full" />
                    ) : (
                      <FileVideo className="w-full h-full" />
                    )}
                  </motion.div>
                </div>

                <div className="space-y-2">
                  <h3 className="text-xl font-semibold text-gray-700">
                    {isDragActive
                      ? "Drop your video here"
                      : "Upload your video"}
                  </h3>
                  <p className="text-gray-500">
                    Drag and drop your video file, or click to browse
                  </p>
                  <p className="text-sm text-gray-400">
                    Supports all video formats - automatically converted to
                    browser-compatible MP4 (max 500MB)
                  </p>
                </div>

                <Button variant="outline" size="lg" className="mt-4">
                  <Upload className="w-4 h-4 mr-2" />
                  Choose File
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl"
          >
            <div className="flex items-center justify-between">
              <p className="text-red-700 text-sm">{error}</p>
              <Button
                variant="ghost"
                size="sm"
                onClick={resetUpload}
                className="text-red-500 hover:text-red-700"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
