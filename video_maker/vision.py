import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from video_maker.models import FaceKeyframe
from video_maker.utils import logger

# ── Tuning constants ──────────────────────────────────────────────────
_FACE_DOWNSCALE_HEIGHT = 480

# Sampling: denser = smoother but slower
_DETECT_INTERVAL = 0.25  # seconds between face detections (4 fps)

# Exponential Moving Average: alpha=0 → no smoothing, alpha=1 → instant
_EMA_ALPHA = 0.25  # low = very smooth "camera operator" feel

# Dead zone: ignore face movements smaller than this (pixels in source res)
_DEAD_ZONE_PX = 40

# Velocity prediction: when face is lost, predict forward by this factor
_VELOCITY_DAMPING = 0.3  # 0 = hold still, 1 = full predicted speed

# Maximum consecutive missed detections before resetting to center
_MAX_MISS_STREAK = 8


def _pick_best_face(
    faces: np.ndarray,
    prev_x: float | None,
    scale: float,
) -> float:
    """Pick the most relevant face from a list of detections.

    Strategy:
    - If we have a previous tracked position, pick the face closest to it
      (temporal continuity — avoids jumping between speakers).
    - Otherwise pick the largest face (most prominent).

    Returns the face center X in original pixel coords.
    """
    if len(faces) == 1 or prev_x is None:
        # Single face or first detection: pick largest
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        return (x + w / 2) / scale

    # Multiple faces + we have history: pick closest to previous position
    best_dist = float("inf")
    best_cx = 0.0
    for (x, y, w, h) in faces:
        cx = (x + w / 2) / scale
        dist = abs(cx - prev_x)
        if dist < best_dist:
            best_dist = dist
            best_cx = cx
    return best_cx


def detect_faces(
    video_path: Path,
    start_time: float,
    end_time: float,
    interval: float = _DETECT_INTERVAL,
) -> List[FaceKeyframe]:
    """Detect faces with smart temporal smoothing for stable crop tracking.

    Pipeline per frame:
    1. Haar detection (histogram-equalized for dark scenes)
    2. Face continuity: pick face closest to previous tracked position
    3. EMA smoothing: smooth_x = alpha * raw_x + (1 - alpha) * prev_smooth_x
    4. Dead zone: skip keyframe if movement < threshold
    5. Velocity prediction: fill gaps when detection misses
    """
    logger.info(
        f"Detecting faces in {video_path.name} "
        f"from {start_time:.1f}s to {end_time:.1f}s (interval={interval}s)"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = min(1.0, _FACE_DOWNSCALE_HEIGHT / height) if height > 0 else 1.0

    keyframes: list[FaceKeyframe] = []
    n_sampled = 0
    n_detected = 0

    # Tracking state
    smooth_x: float | None = None  # EMA-smoothed position
    prev_raw_x: float | None = None  # last raw detection (for velocity)
    velocity: float = 0.0  # pixels per sample interval
    miss_streak = 0  # consecutive frames without detection
    last_emitted_x: float | None = None  # for dead zone

    current_time = start_time

    while current_time <= end_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            current_time += interval
            continue

        n_sampled += 1
        t_rel = current_time - start_time

        # Downscale for speed
        if scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )

        if len(faces) > 0:
            # ── Face detected ──────────────────────────────────────
            n_detected += 1
            raw_x = _pick_best_face(faces, smooth_x, scale)

            # Update velocity (pixels per interval)
            if prev_raw_x is not None:
                velocity = (raw_x - prev_raw_x) * _VELOCITY_DAMPING
            prev_raw_x = raw_x
            miss_streak = 0

            # EMA smoothing
            if smooth_x is None:
                smooth_x = raw_x
            else:
                smooth_x = _EMA_ALPHA * raw_x + (1 - _EMA_ALPHA) * smooth_x

        else:
            # ── No face detected — predict with velocity ───────────
            miss_streak += 1
            if smooth_x is not None and miss_streak <= _MAX_MISS_STREAK:
                smooth_x += velocity
                # Damp velocity over time so we slow down
                velocity *= 0.7
            elif smooth_x is None:
                # Never detected anything yet — skip
                current_time += interval
                continue

        # Clamp to frame bounds
        smooth_x = max(0.0, min(width, smooth_x))

        # Dead zone: only emit keyframe if movement exceeds threshold
        if last_emitted_x is not None and abs(smooth_x - last_emitted_x) < _DEAD_ZONE_PX:
            current_time += interval
            continue

        keyframes.append(FaceKeyframe(time=t_rel, x=int(smooth_x)))
        last_emitted_x = smooth_x
        current_time += interval

    cap.release()

    # Always emit at least first + last timestamps for full coverage
    duration = end_time - start_time
    if keyframes and keyframes[0].time > 0.1:
        keyframes.insert(0, FaceKeyframe(time=0.0, x=keyframes[0].x))
    if keyframes and keyframes[-1].time < duration - 0.5:
        keyframes.append(FaceKeyframe(time=duration, x=keyframes[-1].x))

    det_rate = n_detected / max(n_sampled, 1) * 100
    logger.info(
        f"Face tracking: {n_detected}/{n_sampled} detections ({det_rate:.0f}%), "
        f"{len(keyframes)} keyframes emitted (dead zone filtered)"
    )
    return keyframes


def get_optimal_crop(
    face_keyframes: List[FaceKeyframe],
    src_width: int,
    src_height: int,
) -> Tuple[int, int, int, int]:
    """Calculate the optimal 9:16 crop window based on face positions.
    Returns (crop_w, crop_h, crop_x, crop_y).
    """
    target_ratio = 9 / 16

    crop_h = src_height
    crop_w = int(src_height * target_ratio)

    if crop_w > src_width:
        crop_w = src_width
        crop_h = int(src_width / target_ratio)

    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    if face_keyframes:
        avg_face_x = int(sum(kf.x for kf in face_keyframes) / len(face_keyframes))
    else:
        avg_face_x = src_width // 2

    crop_x = int(max(0, min(src_width - crop_w, avg_face_x - (crop_w // 2))))
    crop_y = int((src_height - crop_h) // 2)

    return crop_w, crop_h, crop_x, crop_y
