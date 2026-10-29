"""Systemcore-equivalent motion layer. Mirrors the WPILib swerve pipeline so the Java port
can be checked against tests/golden/motion_v1.json:

  policy stick -> filter_stick (low-pass + slew limit, stateful, every control period)
  stick -> MathUtil.applyDeadband -> unit-disk clamp -> scale
  -> alliance flip (red: rotate 180 deg, field is rotationally symmetric)
  -> ChassisSpeeds.fromFieldRelativeSpeeds(yaw_estimate)
  -> ChassisSpeeds.discretize(dt)
  -> SwerveDriveKinematics.toSwerveModuleStates -> desaturateWheelSpeeds
  -> SwerveModuleState.optimize(current angle) -> cosineScale(current angle)
"""
from __future__ import annotations

import numpy as np

from .geometry import wrap


def apply_deadband(x, db):
    x = np.clip(x, -1.0, 1.0)
    return np.where(np.abs(x) > db, (x - np.sign(x) * db) / (1.0 - db), 0.0)


def stick_to_field_speeds(stick, alliance, max_v, max_w, deadband):
    """stick (...,3) [forward, left, ccw] alliance-relative -> field ChassisSpeeds (...,3)."""
    s = apply_deadband(np.asarray(stick, float), deadband[..., None] if np.ndim(deadband) else deadband)
    xy = s[..., :2]
    n = np.linalg.norm(xy, axis=-1, keepdims=True)
    xy = xy / np.maximum(n, 1.0) * np.asarray(max_v)[..., None]
    xy = np.where((np.asarray(alliance) == 1)[..., None], -xy, xy)
    return np.concatenate([xy, (s[..., 2] * max_w)[..., None]], -1)


def from_field_relative(speeds, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = speeds[..., 0], speeds[..., 1]
    return np.stack([vx * c + vy * s, -vx * s + vy * c, speeds[..., 2]], -1)


def discretize(speeds, dt):
    """ChassisSpeeds.discretize: twist = Pose2d().log(Pose2d(vx dt, vy dt, w dt)) / dt."""
    dx, dy, dth = speeds[..., 0] * dt, speeds[..., 1] * dt, speeds[..., 2] * dt
    half = dth / 2
    cm1 = np.cos(dth) - 1
    small = np.abs(cm1) < 1e-9
    h = np.where(small, 1.0 - dth ** 2 / 12.0, -(half * np.sin(dth)) / np.where(small, 1.0, cm1))
    norm = np.hypot(h, half)
    c, s = h / norm, -half / norm               # Rotation2d(h, -half)
    tx, ty = (dx * c - dy * s) * norm, (dx * s + dy * c) * norm
    return np.stack([tx / dt, ty / dt, dth / dt], -1)


def to_module_states(speeds, offsets, prev_angle):
    """speeds (...,3) robot-relative, offsets (...,4,2) -> speed (...,4), angle (...,4).
    All-zero chassis speeds keep the previous module angles (WPILib behaviour)."""
    vx, vy, w = speeds[..., 0:1], speeds[..., 1:2], speeds[..., 2:3]
    mx = vx - w * offsets[..., 1]
    my = vy + w * offsets[..., 0]
    spd = np.hypot(mx, my)
    zero = (speeds == 0).all(-1)[..., None]
    ang = np.where(zero, prev_angle, np.arctan2(my, mx))
    return np.where(zero, 0.0, spd), ang


def desaturate(spd, max_speed):
    m = np.abs(spd).max(-1, keepdims=True)
    scale = np.where(m > max_speed[..., None], max_speed[..., None] / np.maximum(m, 1e-12), 1.0)
    return spd * scale


def optimize(spd, ang, current):
    flip = np.abs(wrap(ang - current)) > np.pi / 2
    return np.where(flip, -spd, spd), np.where(flip, wrap(ang + np.pi), wrap(ang))


def cosine_scale(spd, ang, current):
    return spd * np.cos(ang - current)


def filter_stick(cmd, stick, max_v, max_w, tau, accel, alpha, dt):
    """Command filter between the policy and the motion layer, run every control period (robot code, like a WPILib
    LinearFilter + SlewRateLimiter on the joystick). cmd, stick (...,3) alliance-relative [forward, left, ccw] in
    stick units; returns the new cmd. First-order low-pass with time constant tau [s], then the change per period is
    limited to accel [m/s^2] (translation, as a 2D vector) and alpha [rad/s^2] (rotation), converted to stick units
    with the robot's max_v / max_w. Contract for the Java port: tests/golden/cmd_filter_v1.json."""
    cmd, stick = np.asarray(cmd, float), np.clip(np.asarray(stick, float), -1.0, 1.0)
    a = 1.0 - np.exp(-dt / np.asarray(tau, float))
    d = (stick - cmd) * np.asarray(a)[..., None]
    lim_xy = np.asarray(accel, float) * dt / np.asarray(max_v, float)
    n = np.linalg.norm(d[..., :2], axis=-1)
    d_xy = d[..., :2] * np.minimum(1.0, lim_xy / np.maximum(n, 1e-12))[..., None]
    lim_w = np.asarray(alpha, float) * dt / np.asarray(max_w, float)
    d_w = np.clip(d[..., 2], -lim_w, lim_w)
    return cmd + np.concatenate([d_xy, d_w[..., None]], -1)


def motion_layer(stick, alliance, yaw_est, current_angle, prev_target_angle, p, dt):
    """Full pipeline. p: dict of arrays (max_v, max_w, deadband, offsets (…,4,2)).
    Returns module target speed (…,4) [m/s] and angle (…,4) [rad, robot frame]."""
    field = stick_to_field_speeds(stick, alliance, p["max_v"], p["max_w"], p["deadband"])
    robot = discretize(from_field_relative(field, yaw_est), dt)
    spd, ang = to_module_states(robot, p["offsets"], prev_target_angle)
    spd = desaturate(spd, p["max_v"])
    spd, ang = optimize(spd, ang, current_angle)
    return cosine_scale(spd, ang, current_angle), ang
