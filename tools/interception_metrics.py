"""有限球拍盒/球体的纯数值评估；不导入 MuJoCo、文件或绘图代码。

调用方负责状态/参考轨迹的插值，并按时间提供样本。本层只采样评估时窗、
计算几何距离与接近速度门。未规划、无击球时刻和空时窗均保留为未命中；
非法数值及适配器错误向上传播，不能当作已成功完成的失败样本。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


def box_sphere_gap(local, halfsize, radius):
    """球心至有限盒外部的距离减球半径；<=0 表示相交，含边/角擦碰。"""
    local = _vector(local, "local position")
    halfsize = _vector(halfsize, "racket halfsize")
    radius = float(radius)
    if np.any(halfsize <= 0) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("racket halfsize and ball radius must be positive")
    return float(np.linalg.norm(np.maximum(np.abs(local) - halfsize, 0)) - radius)


@dataclass(frozen=True)
class InterceptionProtocol:
    racket_halfsize_m: tuple[float, float, float]
    ball_radius_m: float
    interpolation_step_s: float = .001
    window_before_s: float = .35
    window_after_s: float = .35
    approach_threshold_m_s: float = -.30
    # 保留两个既有入口的端点容差：录像 .5step，消融 .1step。
    endpoint_step_fraction: float = .1

    def __post_init__(self):
        half = _vector(self.racket_halfsize_m, "racket halfsize")
        if np.any(half <= 0):
            raise ValueError("racket halfsize must be positive")
        object.__setattr__(self, "racket_halfsize_m", tuple(float(x) for x in half))
        for name in ("ball_radius_m", "interpolation_step_s", "window_before_s", "window_after_s",
                     "approach_threshold_m_s", "endpoint_step_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.ball_radius_m <= 0 or self.interpolation_step_s <= 0:
            raise ValueError("ball radius and interpolation step must be positive")
        if self.window_before_s < 0 or self.window_after_s < 0:
            raise ValueError("evaluation windows must be nonnegative")
        if not 0 <= self.endpoint_step_fraction <= 1:
            raise ValueError("endpoint step fraction must lie in [0, 1]")

    def as_dict(self):
        return {"schema": "INTERCEPTION_GEOMETRY/1", "racket_halfsize_m": list(self.racket_halfsize_m),
            "ball_radius_m": self.ball_radius_m, "interpolation_step_s": self.interpolation_step_s,
            "window_before_s": self.window_before_s, "window_after_s": self.window_after_s,
            "approach_threshold_m_s": self.approach_threshold_m_s,
            "endpoint_step_fraction": self.endpoint_step_fraction,
            "normal_axis_local": 1, "criterion": "finite_box_sphere_gap<=0",
            "approach_criterion": "relative_normal_velocity<=threshold"}


@dataclass(frozen=True)
class InterceptionSample:
    ball_position_m: object
    ball_velocity_m_s: object
    racket_center_m: object
    world_from_racket: object
    # 可延迟读取：只有几何相交时才需要 MuJoCo 的拍面速度。
    racket_velocity_m_s: object
    planned: bool


@dataclass(frozen=True)
class InterceptionResult:
    contact: bool = False
    approaching: bool = False
    min_gap_m: float | None = None
    first_contact_s: float | None = None
    closest: dict | None = None
    checked_steps: int = 0
    planned_steps: int = 0


def evaluate_interception(sample_at: Callable[[float], InterceptionSample], *,
                          hit_time_s, start_s, end_s, protocol: InterceptionProtocol,
                          maximum_sample_time_s=None, minimum_initial_m=math.inf):
    """在给定记录范围内评估；调用方选择记录起止及是否额外裁切采样端点。

    hit_time_s=None 时仍可记录 checked_steps，但所有样本均不能被判定为命中。
    minimum_initial_m 保留消融旧入口的 1e9 初始化语义，默认不截断最小距离。
    """
    start_s, end_s = float(start_s), float(end_s)
    if not math.isfinite(start_s) or not math.isfinite(end_s):
        raise ValueError("recording bounds must be finite")
    if hit_time_s is not None:
        hit_time_s = float(hit_time_s)
        if not math.isfinite(hit_time_s):
            raise ValueError("hit time must be finite or None")
    if maximum_sample_time_s is not None:
        maximum_sample_time_s = float(maximum_sample_time_s)
        if not math.isfinite(maximum_sample_time_s):
            raise ValueError("sample time limit must be finite")
    minimum_initial_m = float(minimum_initial_m)
    if math.isnan(minimum_initial_m) or minimum_initial_m <= 0:
        raise ValueError("minimum initial distance must be positive")
    begin = max(start_s, hit_time_s - protocol.window_before_s) if hit_time_s is not None else start_s
    finish = min(end_s, hit_time_s + protocol.window_after_s) if hit_time_s is not None else end_s
    if finish < begin:
        return InterceptionResult()
    minimum = minimum_initial_m
    found = approaching = False
    first = closest = None
    checked = planned = 0
    step = protocol.interpolation_step_s
    for now in np.arange(begin, finish + step * protocol.endpoint_step_fraction, step):
        if maximum_sample_time_s is not None and now > maximum_sample_time_s:
            continue
        sample = sample_at(float(now))
        ball = _vector(sample.ball_position_m, "ball position")
        center = _vector(sample.racket_center_m, "racket center")
        rotation = np.asarray(sample.world_from_racket, dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("racket rotation must be a finite 3x3 matrix")
        local = rotation.T @ (ball - center)
        gap = box_sphere_gap(local, protocol.racket_halfsize_m, protocol.ball_radius_m)
        checked += 1
        if not sample.planned or hit_time_s is None:
            continue
        planned += 1
        if gap < minimum:
            minimum = gap
            closest = {"time_s": float(now), "local_m": local.tolist(), "ball_m": ball.tolist(),
                       "racket_center_m": center.tolist()}
        if gap <= 0:
            found = True
            if first is None:
                first = float(now)
            velocity = sample.racket_velocity_m_s
            velocity = velocity() if callable(velocity) else velocity
            relative = _vector(sample.ball_velocity_m_s, "ball velocity") - _vector(velocity, "racket velocity")
            normal = rotation[:, 1] * (1 if local[1] >= 0 else -1)
            if float(relative @ normal) <= protocol.approach_threshold_m_s:
                approaching = True
    return InterceptionResult(found, approaching, None if minimum == minimum_initial_m else float(minimum),
                              first, closest, checked, planned)
