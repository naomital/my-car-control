"""Offline test harness for the depth-estimation algorithm.

Plays back a recorded video, runs Depth Anything V2 on every frame (no
frame dropping here -- this is for evaluating the algorithm, not the live
control loop), and shows three things side by side:

    [ camera frame + steering overlay ]  [ depth map ]

The steering angle is NOT computed by any algorithm yet -- it's driven
entirely by your keyboard (A/D/C), exactly like the live car controls in
connect_wifi_car.py, so you can manually "drive" through the recorded
video and watch how the depth map looks for each steering choice you'd
make. This is meant as a sanity-check tool while the actual
obstacle-avoidance/steering logic is being developed.

Controls
--------
    A        steer left  (-STEERING_STEP_DEG)
    D        steer right (+STEERING_STEP_DEG)
    C        center steering
    SPACE    pause / resume playback
    N        step one frame forward (only while paused)
    Q / ESC  quit

Usage
-----
    python test_depth_steering.py --source path/to/video.mp4
    python test_depth_steering.py --source 0          # webcam, for a quick check
    python test_depth_steering.py --source video.mp4 --input-size 252 --loop
"""

from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np

from depth_runtime import DepthEstimator
from config import CONFIG

# Degree-level steering for this offline visualization -- tune in config.py (CONFIG.steering).
STEERING_MIN_DEG = CONFIG.steering.deg_min
STEERING_MAX_DEG = CONFIG.steering.deg_max
STEERING_STEP_DEG = CONFIG.steering.deg_step

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_steering_overlay(frame: np.ndarray, steering_deg: float, paused: bool) -> np.ndarray:
    """Draws a heading arrow from the bottom-center of the frame plus a
    left<->right gauge bar, so the chosen steering angle is obvious at a
    glance against the depth map next to it."""
    out = frame.copy()
    h, w = out.shape[:2]

    # -- heading arrow --------------------------------------------------
    pivot = (w // 2, h - 10)
    length = int(h * 0.45)
    # 0 deg = straight up; positive = right, negative = left.
    angle_rad = math.radians(steering_deg)
    tip = (
        int(pivot[0] + length * math.sin(angle_rad)),
        int(pivot[1] - length * math.cos(angle_rad)),
    )
    color = (60, 220, 60) if abs(steering_deg) < 1e-6 else (60, 180, 255)
    cv2.arrowedLine(out, pivot, tip, color, 4, tipLength=0.25)

    # -- left/right gauge bar --------------------------------------------
    bar_y = 95
    bar_x0, bar_x1 = 20, w - 20
    cv2.line(out, (bar_x0, bar_y), (bar_x1, bar_y), (200, 200, 200), 2)
    cv2.line(out, (w // 2, bar_y - 8), (w // 2, bar_y + 8), (200, 200, 200), 2)  # center tick
    frac = (steering_deg - STEERING_MIN_DEG) / (STEERING_MAX_DEG - STEERING_MIN_DEG)
    knob_x = int(bar_x0 + frac * (bar_x1 - bar_x0))
    cv2.circle(out, (knob_x, bar_y), 7, color, -1)

    cv2.putText(out, f"steering = {steering_deg:+.0f} deg", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, "A/D steer | C center | SPACE pause | N step | Q quit", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if paused:
        cv2.putText(out, "PAUSED", (w - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 over a recorded video "
                                                   "with keyboard-driven steering overlay.")
    parser.add_argument("--source", required=True, help="Video file path, or an integer (e.g. 0) for a webcam.")
    parser.add_argument("--encoder", default=CONFIG.depth.encoder, choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=CONFIG.depth.input_size,
                         help="Depth model inference resolution (multiple of 14).")
    parser.add_argument("--device", default=CONFIG.depth.device, help="cuda / mps / cpu. Auto-detected if omitted.")
    parser.add_argument("--loop", action="store_true", help="Restart the video automatically when it ends.")
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {args.source}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay_ms = max(1, int(1000.0 / fps_in))

    print("Loading Depth Anything V2 ...")
    estimator = DepthEstimator(encoder=args.encoder, input_size=args.input_size, device=args.device)
    print(f"Depth model ready on device={estimator.device}")

    steering_deg = 0.0
    paused = False

    print("Controls: A/D steer | C center | SPACE pause | N step (while paused) | Q/ESC quit")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("End of video.")
                    break
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                last_frame = frame
                result = estimator.infer_sync(frame)
                last_result = result

            display_frame = draw_steering_overlay(last_frame, steering_deg, paused)
            if last_result.collision_warning:
                cv2.putText(display_frame, "!! OBSTACLE AHEAD !!", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # grayscale (brighter = closer) with the region driving an
            # imminent collision warning painted red; TTC text is already
            # baked in by depth_runtime.
            depth_vis = last_result.visualization.copy()
            cv2.putText(depth_vis, f"depth: {last_result.inference_seconds * 1000:.0f} ms "
                                    f"({1.0 / max(last_result.inference_seconds, 1e-6):.1f} FPS)",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            combined = cv2.hconcat([display_frame, depth_vis])
            cv2.imshow("Depth Anything V2 - steering test (CLICK HERE)", combined)

            key = cv2.waitKey(frame_delay_ms if not paused else 30) & 0xFF
            if key == 255:
                continue

            if key == 27 or key == ord("q"):
                break
            elif key == ord("a"):
                steering_deg = clamp(steering_deg - STEERING_STEP_DEG, STEERING_MIN_DEG, STEERING_MAX_DEG)
            elif key == ord("d"):
                steering_deg = clamp(steering_deg + STEERING_STEP_DEG, STEERING_MIN_DEG, STEERING_MAX_DEG)
            elif key == ord("c"):
                steering_deg = 0.0
            elif key == ord(" "):
                paused = not paused
            elif key == ord("n") and paused:
                ok, frame = cap.read()
                if ok:
                    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    last_frame = frame
                    last_result = estimator.infer_sync(frame)
                elif args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
