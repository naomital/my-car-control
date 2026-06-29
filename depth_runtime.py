"""Real-time Depth Anything V2 inference for the live car camera feed.

Goal: run depth estimation on the frames coming back from the car as fast
and efficiently as possible WITHOUT ever slowing down the drive-control
loop (steering/throttle packets must keep going out at ~20Hz no matter
what the model is doing).

Design
------
- The model runs in its own background thread.
- Only the single most-recent frame matters -- if the model is still busy
  with frame N when frame N+1, N+2, ... arrive, those are simply dropped
  and replaced by the newest one. This is a "latest wins" 1-slot queue,
  not a FIFO, so the depth output is never stale by more than one
  in-flight inference and the GPU/CPU never builds a backlog.
- Smallest checkpoint (vits) + a reduced inference resolution (multiple of
  14, as required by the DINOv2 backbone) for speed. Both are tunable.
- fp16 autocast on CUDA, cudnn.benchmark enabled, eval()/no_grad() always.
- Frame submission and result retrieval are both non-blocking, so the
  caller (the main capture/control loop) never waits on the model.

Visualization + collision warning
----------------------------------
- The depth map is shown as plain grayscale (no colormap): brighter =
  closer, darker = farther. This matches the model's native disparity
  convention (Depth Anything V2 outputs higher-disparity-is-closer).
- There's no metric depth here (monocular, relative-only), so "time to
  collision" is necessarily a heuristic: ``TTCEstimator`` tracks how fast
  the closest surface in the forward region of the frame is filling more
  of the frame's own near/far range, frame over frame, and projects how
  many seconds until it would reach "as close as this frame's own range
  allows". When that projected time drops below a threshold, the pixels
  driving that estimate are highlighted in red on top of the grayscale
  view.

Usage (see connect_wifi_car.py / test_depth_steering.py)
----------------------------------------------------------
    estimator = DepthEstimator()
    estimator.start()
    ...
    estimator.submit_frame(frame_bgr)      # non-blocking, "latest wins"
    result = estimator.get_latest()        # non-blocking, may be None
    if result is not None:
        depth_vis = result.visualization   # uint8 BGR grayscale + red collision overlay
        ttc = result.ttc_seconds           # math.inf if nothing is closing in
    ...
    estimator.stop()
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # car_control/.. -> Topological_nav/
_DA2_DIR = os.path.join(_REPO_ROOT, "models", "Depth-Anything-V2")
if _DA2_DIR not in sys.path:
    sys.path.insert(0, _DA2_DIR)

import torch  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

from config import CONFIG  # noqa: E402

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

DEFAULT_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "checkpoints")


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Time-to-collision heuristic
# =============================================================================

class TTCEstimator:
    """Heuristic, metric-free time-to-collision tracker.

    There's no real distance here -- just relative disparity. So instead
    of seconds-to-a-real-wall, this tracks "closeness" on a 0..1 scale
    (0 = farthest pixel in the current frame's own range, 1 = closest),
    EMA-smooths it, and measures how fast that closeness is rising over a
    short rolling window. Projecting that rate forward to closeness == 1
    gives a relative ETA: useful as an early-warning signal and for
    deciding what to paint red, not as a literal physical countdown.
    """

    def __init__(
        self,
        roi_x_frac: Optional[Tuple[float, float]] = None,
        roi_y_frac: Optional[Tuple[float, float]] = None,
        near_percentile: Optional[float] = None,
        ema_alpha: Optional[float] = None,
        window_seconds: Optional[float] = None,
        danger_ttc_seconds: Optional[float] = None,
        min_closeness_for_warning: Optional[float] = None,
        pixel_highlight_threshold: Optional[float] = None,
    ):
        # Any argument left as None falls back to config.py's TTCConfig
        # defaults (CONFIG.ttc) -- pass explicit values here to override
        # just for one estimator instance without touching the config file.
        cfg = CONFIG.ttc
        self.roi_x_frac = roi_x_frac if roi_x_frac is not None else cfg.roi_x_frac
        self.roi_y_frac = roi_y_frac if roi_y_frac is not None else cfg.roi_y_frac
        self.near_percentile = near_percentile if near_percentile is not None else cfg.near_percentile
        self.ema_alpha = ema_alpha if ema_alpha is not None else cfg.ema_alpha
        self.window_seconds = window_seconds if window_seconds is not None else cfg.window_seconds
        self.danger_ttc_seconds = (
            danger_ttc_seconds if danger_ttc_seconds is not None else cfg.danger_ttc_seconds
        )
        self.min_closeness_for_warning = (
            min_closeness_for_warning if min_closeness_for_warning is not None
            else cfg.min_closeness_for_warning
        )
        self.pixel_highlight_threshold = (
            pixel_highlight_threshold if pixel_highlight_threshold is not None
            else cfg.pixel_highlight_threshold
        )

        self._ema: Optional[float] = None
        self._history: Deque[Tuple[float, float]] = deque()

    def reset(self):
        self._ema = None
        self._history.clear()

    def _roi_slice(self, h: int, w: int):
        y0 = int(h * self.roi_y_frac[0])
        y1 = max(y0 + 1, int(h * self.roi_y_frac[1]))
        x0 = int(w * self.roi_x_frac[0])
        x1 = max(x0 + 1, int(w * self.roi_x_frac[1]))
        return y0, y1, x0, x1

    def update(self, closeness_map: np.ndarray, now: Optional[float] = None) -> "CollisionInfo":
        """``closeness_map`` is float in [0, 1], same shape as the frame,
        0 = farthest pixel this frame, 1 = closest pixel this frame."""
        now = time.perf_counter() if now is None else now
        h, w = closeness_map.shape[:2]
        y0, y1, x0, x1 = self._roi_slice(h, w)
        roi = closeness_map[y0:y1, x0:x1]

        near_closeness = float(np.percentile(roi, self.near_percentile)) if roi.size else 0.0
        self._ema = near_closeness if self._ema is None else (
            (1 - self.ema_alpha) * self._ema + self.ema_alpha * near_closeness
        )

        self._history.append((now, self._ema))
        while self._history and now - self._history[0][0] > self.window_seconds:
            self._history.popleft()

        t_old, ema_old = self._history[0]
        dt = now - t_old
        rate = (self._ema - ema_old) / dt if dt > 1e-3 else 0.0  # closeness/sec, positive = approaching

        if rate > 1e-3:
            ttc = max(0.0, (1.0 - self._ema) / rate)
        else:
            ttc = math.inf

        warning = (
            ttc < self.danger_ttc_seconds
            and self._ema >= self.min_closeness_for_warning
        )

        mask = np.zeros((h, w), dtype=bool)
        if warning:
            full_mask = closeness_map > self.pixel_highlight_threshold
            roi_mask = np.zeros((h, w), dtype=bool)
            roi_mask[y0:y1, x0:x1] = True
            mask = full_mask & roi_mask

        return CollisionInfo(
            ttc_seconds=ttc,
            closing_rate=rate,
            near_closeness=self._ema,
            warning=warning,
            collision_mask=mask,
        )


@dataclass
class CollisionInfo:
    ttc_seconds: float          # math.inf if not currently closing in
    closing_rate: float         # closeness-fraction per second, can be negative (receding)
    near_closeness: float       # EMA-smoothed 0..1 closeness of the nearest forward surface
    warning: bool                # True if ttc_seconds < danger threshold
    collision_mask: np.ndarray   # bool, same size as frame; pixels driving the warning


@dataclass
class DepthResult:
    raw: np.ndarray              # float32 disparity, original frame resolution (higher = closer)
    grayscale: np.ndarray        # uint8 single-channel, brighter = closer, no collision overlay
    visualization: np.ndarray    # uint8 BGR: grayscale + red overlay where collision is detected
    frame_idx: int
    inference_seconds: float
    fps: float
    ttc_seconds: float = field(default=math.inf)
    collision_warning: bool = False


class DepthEstimator:
    """Background thread that keeps producing a depth map (+ collision
    warning) for whatever frame was most recently submitted, as fast as
    the hardware allows."""

    def __init__(
        self,
        encoder: Optional[str] = None,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        input_size: Optional[int] = None,
        device: Optional[str] = None,
        use_half: Optional[bool] = None,
        ttc_estimator: Optional[TTCEstimator] = None,
    ):
        # None-valued args fall back to config.py's DepthModelConfig
        # defaults (CONFIG.depth).
        cfg = CONFIG.depth
        encoder = encoder if encoder is not None else cfg.encoder
        input_size = input_size if input_size is not None else cfg.input_size
        device = device if device is not None else cfg.device
        use_half = use_half if use_half is not None else cfg.use_half

        if input_size % 14 != 0:
            raise ValueError("input_size must be a multiple of 14 (DINOv2 patch size)")

        self.device = resolve_device(device)
        self.input_size = input_size
        self.use_half = (self.device == "cuda") if use_half is None else use_half

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        ckpt_path = os.path.join(checkpoint_dir, f"depth_anything_v2_{encoder}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {ckpt_path}")

        model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model = model.to(self.device).eval()
        # Keep weights in fp32; speed comes from torch.autocast casting the
        # matmul/conv ops to fp16 on the fly (mixing manual .half() weights
        # with autocast is unnecessary and can trip up ops autocast keeps
        # in fp32, e.g. norms).
        self._model = model

        self.ttc = ttc_estimator or TTCEstimator()

        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_idx = 0
        self._latest_result: Optional[DepthResult] = None
        self._frame_counter = 0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # warm up so the first real frame on the car isn't paying for
        # lazy CUDA kernel compilation / cudnn autotuning.
        self._warmup()

    # -- lifecycle ---------------------------------------------------------

    def _warmup(self):
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        try:
            self._infer(dummy, update_ttc=False)
        except Exception:
            pass

    def start(self) -> "DepthEstimator":
        if self._thread is not None:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="depth-estimator")
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- frame in / result out (both non-blocking) -------------------------

    def submit_frame(self, frame_bgr: np.ndarray):
        """Replace whatever frame is currently queued. Never blocks; if the
        worker thread is mid-inference, this frame is simply what it will
        pick up next -- nothing piles up."""
        with self._lock:
            self._pending_frame = frame_bgr
            self._frame_counter += 1
            self._pending_idx = self._frame_counter

    def get_latest(self) -> Optional[DepthResult]:
        with self._lock:
            return self._latest_result

    def infer_sync(self, frame_bgr: np.ndarray) -> DepthResult:
        """Blocking single-frame inference, independent of the background
        thread. Meant for offline/test scripts (e.g. running over a
        recorded video) where every frame should be processed in order
        rather than the "latest wins" live-camera behavior. Still updates
        the same TTC state, so playing a video frame-by-frame gives a
        sensible collision-warning trend."""
        t0 = time.perf_counter()
        depth_raw, grayscale, vis, collision = self._infer(frame_bgr, update_ttc=True)
        dt = time.perf_counter() - t0
        return DepthResult(
            raw=depth_raw, grayscale=grayscale, visualization=vis, frame_idx=-1,
            inference_seconds=dt, fps=1.0 / max(dt, 1e-6),
            ttc_seconds=collision.ttc_seconds, collision_warning=collision.warning,
        )

    # -- worker loop ---------------------------------------------------------

    def _run(self):
        last_done_idx = -1
        last_tick = time.perf_counter()
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._pending_frame
                idx = self._pending_idx
            if frame is None or idx == last_done_idx:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            depth_raw, grayscale, vis, collision = self._infer(frame, update_ttc=True)
            dt = time.perf_counter() - t0

            now = time.perf_counter()
            fps = 1.0 / max(now - last_tick, 1e-6)
            last_tick = now

            result = DepthResult(
                raw=depth_raw, grayscale=grayscale, visualization=vis,
                frame_idx=idx, inference_seconds=dt, fps=fps,
                ttc_seconds=collision.ttc_seconds, collision_warning=collision.warning,
            )
            with self._lock:
                self._latest_result = result
            last_done_idx = idx

    # -- model call + visualization + collision -----------------------------

    @torch.no_grad()
    def _infer(self, frame_bgr: np.ndarray, update_ttc: bool):
        if self.use_half:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth = self._model.infer_image(frame_bgr, self.input_size)
        else:
            depth = self._model.infer_image(frame_bgr, self.input_size)

        depth = np.asarray(depth, dtype=np.float32)
        lo, hi = float(depth.min()), float(depth.max())
        span = max(hi - lo, 1e-6)
        closeness = np.clip((depth - lo) / span, 0.0, 1.0)  # 0..1, 1 = closest
        grayscale = (closeness * 255.0).astype(np.uint8)

        if grayscale.shape[:2] != frame_bgr.shape[:2]:
            grayscale = cv2.resize(grayscale, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
            closeness = cv2.resize(closeness, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

        if update_ttc:
            collision = self.ttc.update(closeness)
        else:
            collision = CollisionInfo(ttc_seconds=math.inf, closing_rate=0.0,
                                       near_closeness=0.0, warning=False,
                                       collision_mask=np.zeros(closeness.shape, dtype=bool))

        vis = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        if collision.warning and collision.collision_mask.any():
            red = np.zeros_like(vis)
            red[:, :] = (0, 0, 255)
            mask3 = collision.collision_mask[:, :, None]
            vis = np.where(mask3, cv2.addWeighted(vis, 0.35, red, 0.65, 0), vis).astype(np.uint8)

        ttc_text = "TTC: --" if math.isinf(collision.ttc_seconds) else f"TTC: {collision.ttc_seconds:.1f}s"
        color = (0, 0, 255) if collision.warning else (255, 255, 255)
        cv2.putText(vis, ttc_text, (vis.shape[1] - 170, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return depth, grayscale, vis, collision
