'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js';

const DEFAULT_THRESHOLD = 70; // Healthy Y-axis pixel threshold between nose and shoulders
const MIN_CONFIDENCE = 0.3; // Minimum confidence score for landmarks

export default function PostureMonitorPage() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const animationFrameIdRef = useRef(null);

  // Status & Telemetry States
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(true);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [postureStatus, setPostureStatus] = useState('calibrating'); // 'good' | 'slouching' | 'calibrating'
  const [liveDistance, setLiveDistance] = useState(null);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [calibrationNotice, setCalibrationNotice] = useState('');
  const [apiSync, setApiSync] = useState({
    status: 'idle',
    lastPayload: null,
    lastTime: null,
    statusCode: null,
  });

  // Mutable refs for tracking in the requestAnimationFrame loop
  const thresholdRef = useRef(threshold);
  useEffect(() => {
    thresholdRef.current = threshold;
  }, [threshold]);

  const lastSentStatusRef = useRef(null);
  const lastSentTimeRef = useRef(0);
  const isSendingRef = useRef(false);
  const latestDistanceRef = useRef(null);

  // POST request to /api/posture
  const sendPostureUpdate = useCallback(async (status) => {
    if (isSendingRef.current) return;
    isSendingRef.current = true;
    lastSentStatusRef.current = status;
    lastSentTimeRef.current = Date.now();

    try {
      const res = await fetch('/api/posture', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status }),
      });
      await res.json().catch(() => null);

      setApiSync({
        status: res.ok ? 'synced' : 'error',
        lastPayload: status,
        lastTime: new Date().toLocaleTimeString(),
        statusCode: res.status,
      });
    } catch (err) {
      console.error('Failed to post posture status to /api/posture:', err);
      setApiSync((prev) => ({
        ...prev,
        status: 'error',
        lastPayload: status,
        lastTime: new Date().toLocaleTimeString(),
      }));
    } finally {
      isSendingRef.current = false;
    }
  }, []);

  // Initialize TensorFlow.js and MoveNet detector
  useEffect(() => {
    let isCancelled = false;

    async function initDetector() {
      try {
        setModelLoading(true);
        setModelError(null);

        // Ensure WebGL backend is initialized
        await tf.setBackend('webgl');
        await tf.ready();

        if (isCancelled) return;

        // Initialize MoveNet SinglePose Lightning detector
        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
            enableSmoothing: true,
          }
        );

        if (isCancelled) {
          detector.dispose?.();
          return;
        }

        detectorRef.current = detector;
        setModelLoading(false);
      } catch (err) {
        console.error('Error loading MoveNet model:', err);
        if (!isCancelled) {
          setModelError('Failed to load posture detection model. Please check WebGL support.');
          setModelLoading(false);
        }
      }
    }

    initDetector();

    return () => {
      isCancelled = true;
      if (detectorRef.current?.dispose) {
        detectorRef.current.dispose();
      }
    };
  }, []);

  // Continuous Pose Detection & Distance Loop
  useEffect(() => {
    let isActive = true;

    const detectPose = async () => {
      if (!isActive) return;

      const video = webcamRef.current?.video;
      const canvas = canvasRef.current;

      if (
        detectorRef.current &&
        video &&
        video.readyState === 4 &&
        canvas &&
        isCameraActive
      ) {
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        if (videoWidth && videoHeight) {
          if (canvas.width !== videoWidth) canvas.width = videoWidth;
          if (canvas.height !== videoHeight) canvas.height = videoHeight;

          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          try {
            const poses = await detectorRef.current.estimatePoses(video);

            if (poses && poses.length > 0 && poses[0].keypoints) {
              const keypoints = poses[0].keypoints;

              const nose = keypoints.find((kp) => kp.name === 'nose');
              const leftShoulder = keypoints.find((kp) => kp.name === 'left_shoulder');
              const rightShoulder = keypoints.find((kp) => kp.name === 'right_shoulder');

              const hasNose = nose && (nose.score ?? 0) >= MIN_CONFIDENCE;
              const hasLeftShoulder =
                leftShoulder && (leftShoulder.score ?? 0) >= MIN_CONFIDENCE;
              const hasRightShoulder =
                rightShoulder && (rightShoulder.score ?? 0) >= MIN_CONFIDENCE;

              if (hasNose && (hasLeftShoulder || hasRightShoulder)) {
                // Calculate shoulder Y and X (averaging if both are visible)
                let shoulderY;
                let shoulderX;

                if (hasLeftShoulder && hasRightShoulder) {
                  shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
                  shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
                } else if (hasLeftShoulder) {
                  shoulderY = leftShoulder.y;
                  shoulderX = leftShoulder.x;
                } else {
                  shoulderY = rightShoulder.y;
                  shoulderX = rightShoulder.x;
                }

                // Y-axis distance between nose and shoulders (shoulders are lower on screen, so higher Y)
                const distanceY = Math.round(shoulderY - nose.y);
                const positiveDistance = Math.max(0, distanceY);

                setLiveDistance(positiveDistance);
                latestDistanceRef.current = positiveDistance;

                // Evaluate posture against healthy threshold
                const currentStatus =
                  positiveDistance < thresholdRef.current ? 'slouching' : 'good';
                setPostureStatus(currentStatus);

                // Send POST request when status changes or periodically (heartbeat every 3s)
                const now = Date.now();
                const statusChanged = currentStatus !== lastSentStatusRef.current;
                const timeSinceLastSend = now - lastSentTimeRef.current;

                if ((statusChanged || timeSinceLastSend >= 3000) && !isSendingRef.current) {
                  sendPostureUpdate(currentStatus);
                }

                // Draw overlay skeleton & measurement graphics
                const strokeColor =
                  currentStatus === 'good' ? 'rgba(52, 211, 153, 0.9)' : 'rgba(244, 63, 94, 0.9)';
                const fillColor =
                  currentStatus === 'good' ? '#10b981' : '#f43f5e';

                ctx.lineWidth = 2;
                ctx.strokeStyle = strokeColor;
                ctx.fillStyle = fillColor;

                // Draw shoulder connection bar if both visible
                if (hasLeftShoulder && hasRightShoulder) {
                  ctx.beginPath();
                  ctx.moveTo(leftShoulder.x, leftShoulder.y);
                  ctx.lineTo(rightShoulder.x, rightShoulder.y);
                  ctx.stroke();
                }

                // Draw vertical distance line between nose and shoulder plane
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(nose.x, nose.y);
                ctx.lineTo(nose.x, shoulderY);
                ctx.stroke();

                // Draw horizontal guide at shoulder level centered at shoulderX
                ctx.beginPath();
                ctx.moveTo(shoulderX - 25, shoulderY);
                ctx.lineTo(shoulderX + 25, shoulderY);
                ctx.stroke();
                ctx.setLineDash([]);

                // Keypoint circles
                // Nose
                ctx.beginPath();
                ctx.arc(nose.x, nose.y, 5, 0, 2 * Math.PI);
                ctx.fill();

                // Left Shoulder
                if (hasLeftShoulder) {
                  ctx.beginPath();
                  ctx.arc(leftShoulder.x, leftShoulder.y, 4, 0, 2 * Math.PI);
                  ctx.fill();
                }

                // Right Shoulder
                if (hasRightShoulder) {
                  ctx.beginPath();
                  ctx.arc(rightShoulder.x, rightShoulder.y, 4, 0, 2 * Math.PI);
                  ctx.fill();
                }

                // Measurement label next to vertical line
                ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
                ctx.fillStyle = '#ffffff';
                ctx.fillText(`${positiveDistance}px`, nose.x + 10, (nose.y + shoulderY) / 2);
              } else {
                setPostureStatus('calibrating');
              }
            } else {
              setPostureStatus('calibrating');
            }
          } catch (loopErr) {
            console.error('Frame pose estimation error:', loopErr);
          }
        }
      }

      if (isActive) {
        animationFrameIdRef.current = requestAnimationFrame(detectPose);
      }
    };

    if (!modelLoading && isCameraActive) {
      animationFrameIdRef.current = requestAnimationFrame(detectPose);
    }

    return () => {
      isActive = false;
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, [modelLoading, isCameraActive, sendPostureUpdate]);

  // Calibration function: Calibrate healthy threshold based on current upright posture
  const handleCalibrate = () => {
    if (latestDistanceRef.current && latestDistanceRef.current > 30) {
      // Set healthy threshold to 82% of current upright measurement
      const calibratedThreshold = Math.max(35, Math.round(latestDistanceRef.current * 0.82));
      setThreshold(calibratedThreshold);
      setCalibrationNotice(
        `Calibrated to ${calibratedThreshold}px (baseline: ${latestDistanceRef.current}px)`
      );
      setTimeout(() => setCalibrationNotice(''), 4000);
    } else {
      setCalibrationNotice('Could not calibrate: please sit upright facing the camera.');
      setTimeout(() => setCalibrationNotice(''), 4000);
    }
  };

  const handleResetThreshold = () => {
    setThreshold(DEFAULT_THRESHOLD);
    setCalibrationNotice(`Reset to default (${DEFAULT_THRESHOLD}px)`);
    setTimeout(() => setCalibrationNotice(''), 3000);
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col items-center justify-between p-4 sm:p-8 font-sans antialiased selection:bg-zinc-800">
      {/* Header Bar */}
      <header className="w-full max-w-4xl flex items-center justify-between pb-6 border-b border-zinc-800/80">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 ring-4 ring-emerald-500/20" />
          <div className="flex flex-col">
            <span className="text-sm font-semibold tracking-wider uppercase text-zinc-200">
              ALIGN // Sentinel
            </span>
            <span className="text-xs text-zinc-500 tracking-tight">
              MoveNet Realtime Posture Tracking
            </span>
          </div>
        </div>

        {/* API Endpoint Sync Status */}
        <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              apiSync.status === 'synced'
                ? 'bg-emerald-400'
                : apiSync.status === 'error'
                ? 'bg-rose-500'
                : 'bg-zinc-500'
            }`}
          />
          <span className="font-mono text-zinc-400">/api/posture</span>
          {apiSync.lastPayload && (
            <span className="font-mono text-zinc-300 font-medium">
              [{apiSync.lastPayload}]
            </span>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="w-full max-w-4xl flex flex-col items-center gap-6 my-auto py-6">
        {/* Status Indicator Card */}
        <div className="w-full flex flex-col items-center text-center">
          <div
            className={`inline-flex items-center gap-3 px-5 py-2.5 rounded-full border transition-all duration-300 ${
              postureStatus === 'good'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : postureStatus === 'slouching'
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                : 'bg-zinc-900 border-zinc-800 text-zinc-400'
            }`}
          >
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                postureStatus === 'good'
                  ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]'
                  : postureStatus === 'slouching'
                  ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)] animate-pulse'
                  : 'bg-zinc-500 animate-pulse'
              }`}
            />
            <span className="text-xs sm:text-sm font-semibold tracking-widest uppercase">
              {postureStatus === 'good'
                ? 'Optimal Posture'
                : postureStatus === 'slouching'
                ? 'Slouch Detected'
                : 'Aligning Sensor...'}
            </span>
          </div>

          <p className="text-xs text-zinc-400 mt-2.5 max-w-sm">
            {postureStatus === 'good'
              ? 'Spine and cervical distance within healthy upright alignment.'
              : postureStatus === 'slouching'
              ? 'Nose-to-shoulder vertical distance is below threshold. Lift your chest.'
              : 'Position your face and shoulders within clear camera view.'}
          </p>
        </div>

        {/* Centered Webcam Feed with Visual Overlay */}
        <div className="relative w-full max-w-xl aspect-[4/3] rounded-2xl overflow-hidden bg-zinc-900 border border-zinc-800/90 shadow-2xl">
          {modelLoading && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-zinc-950/90 gap-3 backdrop-blur-sm">
              <div className="w-8 h-8 rounded-full border-2 border-zinc-700 border-t-zinc-200 animate-spin" />
              <p className="text-xs font-mono tracking-wide text-zinc-400">
                Loading MoveNet Model (WebGL)...
              </p>
            </div>
          )}

          {modelError && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-zinc-950/95 p-6 text-center">
              <p className="text-xs text-rose-400 font-medium mb-2">{modelError}</p>
              <button
                onClick={() => window.location.reload()}
                className="text-xs px-3 py-1.5 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-200"
              >
                Reload Application
              </button>
            </div>
          )}

          {isCameraActive ? (
            <>
              <Webcam
                ref={webcamRef}
                audio={false}
                mirrored={true}
                screenshotFormat="image/jpeg"
                videoConstraints={{
                  width: 640,
                  height: 480,
                  facingMode: 'user',
                }}
                onUserMedia={() => setIsCameraReady(true)}
                className="w-full h-full object-cover"
              />
              {/* Canvas Overlay for Pose Landmarks - mirrored to align with mirrored webcam */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-cover pointer-events-none -scale-x-100"
              />
              {!isCameraReady && !modelLoading && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-zinc-950/60 backdrop-blur-sm">
                  <span className="text-xs font-mono text-zinc-400 animate-pulse">
                    Connecting camera feed...
                  </span>
                </div>
              )}
            </>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-zinc-600 gap-2">
              <svg
                className="w-10 h-10"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="1.5"
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <span className="text-xs tracking-wider uppercase font-mono">Camera Paused</span>
            </div>
          )}

          {/* Top-Right HUD Tag */}
          <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-md bg-zinc-950/70 border border-zinc-800 text-[10px] font-mono uppercase tracking-wider text-zinc-400 backdrop-blur-sm">
              MoveNet Lightning
            </span>
          </div>

          {/* Bottom Feed Overlay Stats */}
          <div className="absolute bottom-3 inset-x-3 z-10 flex items-center justify-between text-[11px] font-mono text-zinc-300">
            <div className="px-2.5 py-1 rounded-md bg-zinc-950/80 border border-zinc-800/80 backdrop-blur-sm">
              Delta:{' '}
              <span
                className={`font-semibold ${
                  postureStatus === 'good'
                    ? 'text-emerald-400'
                    : postureStatus === 'slouching'
                    ? 'text-rose-400'
                    : 'text-zinc-400'
                }`}
              >
                {liveDistance !== null ? `${liveDistance}px` : '--'}
              </span>
            </div>

            <div className="px-2.5 py-1 rounded-md bg-zinc-950/80 border border-zinc-800/80 backdrop-blur-sm text-zinc-400">
              Threshold: <span className="text-zinc-200">{threshold}px</span>
            </div>
          </div>
        </div>

        {/* Telemetry Metrics Bar */}
        <div className="w-full max-w-xl grid grid-cols-3 gap-3">
          <div className="flex flex-col p-3 rounded-xl bg-zinc-900/60 border border-zinc-800/80">
            <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500">
              Distance (Y)
            </span>
            <span className="text-lg font-mono font-semibold text-zinc-100 mt-0.5">
              {liveDistance !== null ? `${liveDistance} px` : '--'}
            </span>
            <span className="text-[10px] text-zinc-500 mt-1">Nose to shoulders</span>
          </div>

          <div className="flex flex-col p-3 rounded-xl bg-zinc-900/60 border border-zinc-800/80">
            <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500">
              Threshold
            </span>
            <span className="text-lg font-mono font-semibold text-zinc-100 mt-0.5">
              {threshold} px
            </span>
            <span className="text-[10px] text-zinc-500 mt-1">Slouch trigger level</span>
          </div>

          <div className="flex flex-col p-3 rounded-xl bg-zinc-900/60 border border-zinc-800/80">
            <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500">
              POST /api/posture
            </span>
            <span
              className={`text-lg font-mono font-semibold mt-0.5 capitalize ${
                apiSync.status === 'synced'
                  ? 'text-emerald-400'
                  : apiSync.status === 'error'
                  ? 'text-rose-400'
                  : 'text-zinc-400'
              }`}
            >
              {apiSync.lastPayload || 'Idle'}
            </span>
            <span className="text-[10px] text-zinc-500 mt-1">
              {apiSync.lastTime ? `Synced ${apiSync.lastTime}` : 'Awaiting data'}
            </span>
          </div>
        </div>

        {/* Controls Panel */}
        <div className="w-full max-w-xl flex flex-col gap-4 p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/70">
          <div className="flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-xs font-medium text-zinc-200">
                Posture Threshold Calibration
              </span>
              <span className="text-[11px] text-zinc-500">
                Adjust sensitivity or calibrate to your upright height
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleCalibrate}
                className="px-3 py-1.5 rounded-lg bg-zinc-100 hover:bg-white text-zinc-900 text-xs font-medium transition-colors cursor-pointer"
              >
                Calibrate Current
              </button>
              <button
                type="button"
                onClick={handleResetThreshold}
                className="px-2.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs transition-colors cursor-pointer"
                title="Reset to default (70px)"
              >
                Reset
              </button>
              <button
                type="button"
                onClick={() => setIsCameraActive((prev) => !prev)}
                className="px-2.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs transition-colors cursor-pointer"
                title={isCameraActive ? 'Pause Camera' : 'Resume Camera'}
              >
                {isCameraActive ? 'Pause' : 'Resume'}
              </button>
            </div>
          </div>

          {/* Threshold Slider */}
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-mono text-zinc-500 w-8">30px</span>
            <input
              type="range"
              min="30"
              max="130"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="flex-1 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-zinc-200"
            />
            <span className="text-[10px] font-mono text-zinc-500 w-10 text-right">
              130px
            </span>
          </div>

          {calibrationNotice && (
            <p className="text-xs font-mono text-emerald-400/90 text-center animate-fade-in">
              {calibrationNotice}
            </p>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="w-full max-w-4xl pt-4 border-t border-zinc-900 flex items-center justify-between text-[11px] text-zinc-600 font-mono">
        <span>TENSORFLOW.JS // MOVENET LIGHTNING</span>
        <span>NEXT.js CLIENT COMPONENT</span>
      </footer>
    </div>
  );
}
