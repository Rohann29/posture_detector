'use client';

import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js';

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
  
  // Scale-invariant posture metrics
  const [liveRatio, setLiveRatio] = useState(null);
  const [liveDistance, setLiveDistance] = useState(null);
  const [liveShoulderWidth, setLiveShoulderWidth] = useState(null);
  const [baseline, setBaseline] = useState(null); // Calibrated baseline postureRatio
  const [calibrationNotice, setCalibrationNotice] = useState('');
  
  const [apiSync, setApiSync] = useState({
    status: 'idle',
    lastPayload: null,
    lastTime: null,
    statusCode: null,
  });

  // Mutable refs for tracking within animation frames without stale closures
  const baselineRef = useRef(baseline);
  useEffect(() => {
    baselineRef.current = baseline;
  }, [baseline]);

  const latestRatioRef = useRef(null);

  // 1. THE API BRIDGE: Dedicated useEffect watching postureStatus
  // Whenever the state changes (and ONLY when it changes), fire POST to /api/posture
  useEffect(() => {
    if (postureStatus !== 'good' && postureStatus !== 'slouching') {
      return;
    }

    const currentStatus = postureStatus;

    async function firePostureApi() {
      try {
        const res = await fetch('/api/posture', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ status: currentStatus }),
        });

        await res.json().catch(() => null);

        setApiSync({
          status: res.ok ? 'synced' : 'error',
          lastPayload: currentStatus,
          lastTime: new Date().toLocaleTimeString(),
          statusCode: res.status,
        });
      } catch (err) {
        console.error('Failed to post posture status to /api/posture:', err);
        setApiSync((prev) => ({
          ...prev,
          status: 'error',
          lastPayload: currentStatus,
          lastTime: new Date().toLocaleTimeString(),
        }));
      }
    }

    firePostureApi();
  }, [postureStatus]);

  // Page Visibility API listener: silence buzzer alarm when switching tabs
  useEffect(() => {
    const handleVisibilityChange = async () => {
      if (document.hidden) {
        // Tab inactive: immediately silence database alarm by posting {"status": "good"}
        setPostureStatus('Paused (Tab Inactive)');

        try {
          const res = await fetch('/api/posture', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ status: 'good' }),
          });

          await res.json().catch(() => null);

          setApiSync({
            status: res.ok ? 'synced' : 'error',
            lastPayload: 'good (tab inactive)',
            lastTime: new Date().toLocaleTimeString(),
            statusCode: res.status,
          });
        } catch (err) {
          console.error('Failed to silence posture alarm on visibilitychange:', err);
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
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

  // Continuous Pose Detection Loop with Scale-Invariant Math
  useEffect(() => {
    let isActive = true;

    const detectPose = async () => {
      if (!isActive) return;

      const video = webcamRef.current?.video;
      const canvas = canvasRef.current;

      // If document is hidden (user switched tabs), pause pose tracking loop until tab is active again
      if (typeof document !== 'undefined' && document.hidden) {
        if (isActive) {
          animationFrameIdRef.current = requestAnimationFrame(detectPose);
        }
        return;
      }

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

            // 1. If poses.length === 0 (no person detected), set UI status to "No person detected"
            if (!poses || poses.length === 0) {
              setPostureStatus('No person detected');
              setLiveRatio(null);
              setLiveDistance(null);
              setLiveShoulderWidth(null);

              // 2. CRITICAL: Still call requestAnimationFrame at the end of this block to keep loop running
              if (isActive) {
                animationFrameIdRef.current = requestAnimationFrame(detectPose);
              }
              return;
            }

            if (poses[0].keypoints) {
              const keypoints = poses[0].keypoints;

              const nose = keypoints.find((kp) => kp.name === 'nose');
              const leftShoulder = keypoints.find((kp) => kp.name === 'left_shoulder');
              const rightShoulder = keypoints.find((kp) => kp.name === 'right_shoulder');

              const hasNose = nose && (nose.score ?? 0) >= MIN_CONFIDENCE;
              const hasLeftShoulder =
                leftShoulder && (leftShoulder.score ?? 0) >= MIN_CONFIDENCE;
              const hasRightShoulder =
                rightShoulder && (rightShoulder.score ?? 0) >= MIN_CONFIDENCE;

              if (hasNose && hasLeftShoulder && hasRightShoulder) {
                // 1. Calculate shoulderWidth: distance between left_shoulder.x and right_shoulder.x
                const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x);

                // 2. Calculate verticalDistance between nose.y and midpoint of shoulders
                const midpointShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
                const midpointShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
                const verticalDistance = midpointShoulderY - nose.y;

                if (shoulderWidth > 20) {
                  // 3. Calculate postureRatio = verticalDistance / shoulderWidth
                  const postureRatio = Math.max(0, verticalDistance) / shoulderWidth;

                  latestRatioRef.current = postureRatio;
                  setLiveRatio(postureRatio);
                  setLiveDistance(Math.round(verticalDistance));
                  setLiveShoulderWidth(Math.round(shoulderWidth));

                  // 5. Trigger "slouching" status if live postureRatio drops below 85% of calibrated baseline
                  if (baselineRef.current !== null) {
                    const slouchThreshold = baselineRef.current * 0.85;
                    const nextStatus = postureRatio < slouchThreshold ? 'slouching' : 'good';
                    setPostureStatus(nextStatus);
                  } else {
                    setPostureStatus('calibrating');
                  }

                  // Draw clean visual overlay on canvas
                  const isGood =
                    baselineRef.current !== null
                      ? postureRatio >= baselineRef.current * 0.85
                      : true;

                  const strokeColor = isGood
                    ? 'rgba(52, 211, 153, 0.9)'
                    : 'rgba(244, 63, 94, 0.9)';
                  const fillColor = isGood ? '#10b981' : '#f43f5e';

                  ctx.lineWidth = 2;
                  ctx.strokeStyle = strokeColor;
                  ctx.fillStyle = fillColor;

                  // Shoulder span horizontal line (shoulderWidth)
                  ctx.beginPath();
                  ctx.moveTo(leftShoulder.x, leftShoulder.y);
                  ctx.lineTo(rightShoulder.x, rightShoulder.y);
                  ctx.stroke();

                  // Vertical line from nose to shoulder midpoint (verticalDistance)
                  ctx.setLineDash([4, 4]);
                  ctx.beginPath();
                  ctx.moveTo(nose.x, nose.y);
                  ctx.lineTo(midpointShoulderX, midpointShoulderY);
                  ctx.stroke();
                  ctx.setLineDash([]);

                  // Keypoint dots
                  // Nose
                  ctx.beginPath();
                  ctx.arc(nose.x, nose.y, 5, 0, 2 * Math.PI);
                  ctx.fill();

                  // Left & Right Shoulder
                  ctx.beginPath();
                  ctx.arc(leftShoulder.x, leftShoulder.y, 4, 0, 2 * Math.PI);
                  ctx.fill();

                  ctx.beginPath();
                  ctx.arc(rightShoulder.x, rightShoulder.y, 4, 0, 2 * Math.PI);
                  ctx.fill();

                  // Shoulder Midpoint dot
                  ctx.beginPath();
                  ctx.arc(midpointShoulderX, midpointShoulderY, 3, 0, 2 * Math.PI);
                  ctx.fill();

                  // Metric label overlay next to vertical guide line
                  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
                  ctx.fillStyle = '#ffffff';
                  ctx.fillText(
                    `Ratio: ${postureRatio.toFixed(2)}`,
                    midpointShoulderX + 12,
                    (nose.y + midpointShoulderY) / 2
                  );
                }
              } else {
                setPostureStatus('No person detected');
                setLiveRatio(null);
                setLiveDistance(null);
                setLiveShoulderWidth(null);
              }
            } else {
              setPostureStatus('No person detected');
              setLiveRatio(null);
              setLiveDistance(null);
              setLiveShoulderWidth(null);
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
  }, [modelLoading, isCameraActive]);

  // 4. THE ACCURACY FIX: Calibrate Baseline Handler
  // Saves the current scale-invariant postureRatio as the baseline state variable
  const handleCalibrateBaseline = () => {
    if (latestRatioRef.current !== null && latestRatioRef.current > 0.05) {
      const newBaseline = latestRatioRef.current;
      setBaseline(newBaseline);
      baselineRef.current = newBaseline;
      setPostureStatus('good');

      const slouchThreshold = (newBaseline * 0.85).toFixed(2);
      setCalibrationNotice(
        `Baseline set: ${newBaseline.toFixed(2)} (Slouch threshold: < ${slouchThreshold})`
      );
      setTimeout(() => setCalibrationNotice(''), 4500);
    } else {
      setCalibrationNotice(
        'Calibration failed: Please position your face and shoulders clearly in camera.'
      );
      setTimeout(() => setCalibrationNotice(''), 4000);
    }
  };

  const handleResetBaseline = () => {
    setBaseline(null);
    baselineRef.current = null;
    setPostureStatus('calibrating');
    setCalibrationNotice('Baseline cleared. Click "Calibrate Baseline" to re-calibrate.');
    setTimeout(() => setCalibrationNotice(''), 3500);
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
              Scale-Invariant Posture Ratio Tracking
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
                : postureStatus === 'No person detected'
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                : postureStatus === 'Paused (Tab Inactive)'
                ? 'bg-zinc-800/80 border-zinc-700/80 text-zinc-300'
                : 'bg-zinc-900 border-zinc-800 text-zinc-400'
            }`}
          >
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                postureStatus === 'good'
                  ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]'
                  : postureStatus === 'slouching'
                  ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)] animate-pulse'
                  : postureStatus === 'No person detected'
                  ? 'bg-amber-400 animate-pulse'
                  : postureStatus === 'Paused (Tab Inactive)'
                  ? 'bg-zinc-400'
                  : 'bg-zinc-500 animate-pulse'
              }`}
            />
            <span className="text-xs sm:text-sm font-semibold tracking-widest uppercase">
              {postureStatus === 'good'
                ? 'Optimal Posture'
                : postureStatus === 'slouching'
                ? 'Slouch Detected'
                : postureStatus === 'No person detected'
                ? 'No person detected'
                : postureStatus === 'Paused (Tab Inactive)'
                ? 'Paused (Tab Inactive)'
                : baseline === null
                ? 'Calibration Required'
                : 'Aligning Sensor...'}
            </span>
          </div>

          <p className="text-xs text-zinc-400 mt-2.5 max-w-md">
            {postureStatus === 'good'
              ? 'Spine and cervical distance within healthy upright alignment.'
              : postureStatus === 'slouching'
              ? 'Live ratio dropped below 85% of your baseline. Straighten up.'
              : postureStatus === 'No person detected'
              ? 'No person detected in camera view. Step into frame to resume.'
              : postureStatus === 'Paused (Tab Inactive)'
              ? 'Tab inactive. Alarm silenced to good. Return to tab to resume.'
              : baseline === null
              ? 'Sit upright facing the camera and click "Calibrate Baseline" below.'
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
                type="button"
                onClick={() => window.location.reload()}
                className="text-xs px-3 py-1.5 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-200 cursor-pointer"
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
              Scale Invariant
            </span>
          </div>

          {/* Bottom Feed Overlay Stats */}
          <div className="absolute bottom-3 inset-x-3 z-10 flex items-center justify-between text-[11px] font-mono text-zinc-300">
            <div className="px-2.5 py-1 rounded-md bg-zinc-950/80 border border-zinc-800/80 backdrop-blur-sm">
              Live Ratio:{' '}
              <span
                className={`font-semibold ${
                  postureStatus === 'good'
                    ? 'text-emerald-400'
                    : postureStatus === 'slouching'
                    ? 'text-rose-400'
                    : 'text-zinc-400'
                }`}
              >
                {liveRatio !== null ? liveRatio.toFixed(2) : '--'}
              </span>
            </div>

            <div className="px-2.5 py-1 rounded-md bg-zinc-950/80 border border-zinc-800/80 backdrop-blur-sm text-zinc-400">
              Baseline:{' '}
              <span className="text-zinc-200">
                {baseline !== null ? baseline.toFixed(2) : 'Unset'}
              </span>
            </div>
          </div>
        </div>

        {/* Telemetry Metrics Bar */}
        <div className="w-full max-w-xl grid grid-cols-3 gap-3">
          <div className="flex flex-col p-3 rounded-xl bg-zinc-900/60 border border-zinc-800/80">
            <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500">
              Posture Ratio
            </span>
            <span className="text-lg font-mono font-semibold text-zinc-100 mt-0.5">
              {liveRatio !== null ? liveRatio.toFixed(2) : '--'}
            </span>
            <span className="text-[10px] text-zinc-500 mt-1">
              {liveDistance !== null && liveShoulderWidth !== null
                ? `${liveDistance}px / ${liveShoulderWidth}px`
                : 'vertical / width'}
            </span>
          </div>

          <div className="flex flex-col p-3 rounded-xl bg-zinc-900/60 border border-zinc-800/80">
            <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500">
              Baseline / Threshold
            </span>
            <span className="text-lg font-mono font-semibold text-zinc-100 mt-0.5">
              {baseline !== null ? (
                <>
                  {baseline.toFixed(2)}{' '}
                  <span className="text-xs text-zinc-400 font-normal">
                    (&lt;{(baseline * 0.85).toFixed(2)})
                  </span>
                </>
              ) : (
                'Uncalibrated'
              )}
            </span>
            <span className="text-[10px] text-zinc-500 mt-1">Slouch at &lt; 85% baseline</span>
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
              {apiSync.lastPayload || (postureStatus !== 'calibrating' ? postureStatus : 'Idle')}
            </span>
            <span className="text-[10px] text-zinc-500 mt-1">
              {apiSync.lastTime ? `Synced ${apiSync.lastTime}` : 'Awaiting change'}
            </span>
          </div>
        </div>

        {/* Controls & Calibration Panel */}
        <div className="w-full max-w-xl flex flex-col gap-3 p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/70">
          <div className="flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-xs font-medium text-zinc-200">
                Baseline Calibration
              </span>
              <span className="text-[11px] text-zinc-500">
                Calibrates scale-invariant ratio (vertical distance / shoulder width)
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleCalibrateBaseline}
                className="px-3 py-1.5 rounded-lg bg-zinc-100 hover:bg-white text-zinc-900 text-xs font-semibold transition-colors cursor-pointer shadow-sm active:scale-95"
              >
                Calibrate Baseline
              </button>

              {baseline !== null && (
                <button
                  type="button"
                  onClick={handleResetBaseline}
                  className="px-2.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs transition-colors cursor-pointer"
                  title="Clear baseline"
                >
                  Reset
                </button>
              )}

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

          {calibrationNotice && (
            <p className="text-xs font-mono text-emerald-400/90 text-center animate-fade-in mt-1">
              {calibrationNotice}
            </p>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="w-full max-w-4xl pt-4 border-t border-zinc-900 flex items-center justify-between text-[11px] text-zinc-600 font-mono">
        <span>TENSORFLOW.JS // MOVENET LIGHTNING</span>
        <span>SCALE-INVARIANT RATIO TELEMETRY</span>
      </footer>
    </div>
  );
}
