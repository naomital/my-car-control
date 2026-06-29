"""Central tunable configuration for the car control + depth pipeline.

Everything that's a "tune this number" knob lives here, in one place:
steering, throttle, the depth model itself, and the TTC (time-to-collision)
collision-warning heuristic. connect_wifi_car.py, test_depth_steering.py
and depth_runtime.py all read their defaults from ``CONFIG`` below -- edit
this file to retune behavior instead of hunting through each script.

Each script can still override any individual value via its own CLI flags
or constructor args; these are just the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SteeringConfig:
    """Byte-level values are what's actually sent over UDP to the car
    (connect_wifi_car.py). Degree-level values are only used by the
    keyboard-driven steering overlay in the offline test script
    (test_depth_steering.py) -- there's no real car to steer there, just a
    visual heading indicator."""

    # -- byte protocol (0x00-0xFF, center = 0x80) ------------------------
    center: int = 0x80
    min: int = 0x40
    max: int = 0xC0
    step: int = 16

    # -- degrees, for the offline video test's on-screen arrow/gauge -----
    deg_min: float = -45.0
    deg_max: float = 45.0
    deg_step: float = 10.0


@dataclass
class ThrottleConfig:
    """Byte-level throttle values sent over UDP (connect_wifi_car.py only)."""

    stop: int = 0x80
    forward: int = 0x94
    backward: int = 0x64


@dataclass
class DepthModelConfig:
    """Which Depth Anything V2 checkpoint to run and how, for speed."""

    encoder: str = "vits"            # vits = smallest/fastest; vitb/vitl/vitg are larger+slower
    input_size: int = 252            # inference resolution, must be a multiple of 14
    device: Optional[str] = None     # None = auto-detect (cuda > mps > cpu)
    use_half: Optional[bool] = None  # None = auto (fp16 autocast on cuda, fp32 elsewhere)


@dataclass
class TTCConfig:
    """Tuning for the heuristic, metric-free time-to-collision estimator
    (see TTCEstimator in depth_runtime.py for the full explanation of each
    knob). There is no real distance here -- "closeness" is 0..1 relative
    to each frame's own near/far range, and TTC is the projected time
    until that closeness would reach 1 (i.e. "as close as this frame's
    own range allows") at the current closing rate."""

    # Region of the frame treated as "the path ahead" when looking for
    # the nearest obstacle, as a fraction of (width, height). Defaults to
    # the center 40% of the width and the top 75% of the height (skips
    # the bottom strip, which is mostly ground right in front of the
    # camera and would otherwise look "always close").
    roi_x_frac: Tuple[float, float] = (0.30, 0.70)
    roi_y_frac: Tuple[float, float] = (0.0, 0.75)

    near_percentile: float = 90.0          # percentile used to pick "the nearest surface" within the ROI
    ema_alpha: float = 0.3                 # smoothing on the near-closeness signal (higher = more reactive)
    window_seconds: float = 0.5            # rolling window used to estimate the closing rate
    danger_ttc_seconds: float = 3.0        # warn when projected TTC drops below this (raised -> warns earlier)
    min_closeness_for_warning: float = 0.35  # ignore closing-rate noise far away (lowered -> triggers sooner)
    pixel_highlight_threshold: float = 0.50  # how close (0..1) a pixel must be to get painted red (lowered -> earlier)


@dataclass
class NavigationConfig:
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    throttle: ThrottleConfig = field(default_factory=ThrottleConfig)
    depth: DepthModelConfig = field(default_factory=DepthModelConfig)
    ttc: TTCConfig = field(default_factory=TTCConfig)


# Import this from any script: `from config import CONFIG`
CONFIG = NavigationConfig()
