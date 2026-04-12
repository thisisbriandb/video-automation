import cv2
import numpy as np
import urllib.request
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

# Maximum consecutive missed detections before holding position
_MAX_MISS_STREAK = 8

# Camera inertia: max pixels the smoothed position can move per sample interval
# Prevents sudden jumps even when EMA target changes abruptly
_MAX_SPEED_PX = 80  # 80px per 0.25s = 320px/s ≈ full pan in ~6s


# ── DNN Face Detector (YuNet — much more stable than Haar) ───────────
_MODELS_DIR = Path(__file__).parent.parent / "models"
_YUNET_PATH = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_HAS_YUNET_API = hasattr(cv2, "FaceDetectorYN")
_yunet_checked = False
_yunet_ok = False


def _ensure_yunet_model() -> bool:
    """Download YuNet ONNX model on first use (~400 KB). Returns True if available."""
    global _yunet_checked, _yunet_ok
    if _yunet_checked:
        return _yunet_ok
    _yunet_checked = True

    if not _HAS_YUNET_API:
        logger.info("OpenCV too old for FaceDetectorYN — using Haar cascade")
        return False

    if _YUNET_PATH.exists():
        _yunet_ok = True
        return True

    try:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading YuNet face detection model (~400 KB)...")
        urllib.request.urlretrieve(_YUNET_URL, str(_YUNET_PATH))
        _yunet_ok = True
        logger.info(f"YuNet model saved: {_YUNET_PATH}")
    except Exception as e:
        logger.warning(f"YuNet download failed ({e}) — falling back to Haar cascade")
    return _yunet_ok


# ── Per-frame detection ──────────────────────────────────────────────

def _detect_in_frame(
    frame_bgr: np.ndarray,
    yunet: "cv2.FaceDetectorYN | None",
    haar: "cv2.CascadeClassifier | None",
) -> list[tuple[int, int, int, int]]:
    """Detect faces in one frame. Returns [(x, y, w, h), ...] in frame coords."""
    if yunet is not None:
        _, raw = yunet.detect(frame_bgr)
        if raw is not None and len(raw) > 0:
            return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in raw]
        return []

    # Haar fallback (needs grayscale + histogram eq)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = haar.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces] if len(faces) > 0 else []


def _pick_best_face(
    faces: list[tuple[int, int, int, int]],
    prev_x: float | None,
    scale: float,
) -> float:
    """Pick the most relevant face (temporal continuity > size).

    - If we have a previous tracked position, pick the face closest to it
      (avoids jumping between speakers).
    - Otherwise pick the largest face (most prominent).

    Returns the face center X in original pixel coords.
    """
    if len(faces) == 1 or prev_x is None:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        return (x + w / 2) / scale

    best_dist = float("inf")
    best_cx = 0.0
    for (x, y, w, h) in faces:
        cx = (x + w / 2) / scale
        dist = abs(cx - prev_x)
        if dist < best_dist:
            best_dist = dist
            best_cx = cx
    return best_cx


# ── Main tracking loop ───────────────────────────────────────────────

def detect_faces(
    video_path: Path,
    start_time: float,
    end_time: float,
    interval: float = _DETECT_INTERVAL,
) -> List[FaceKeyframe]:
    """Detect faces with DNN + EMA smoothing + speed cap for stable crop tracking.

    Pipeline per sample:
    1. DNN detection (YuNet) with Haar cascade fallback
    2. Face continuity: pick face closest to previous tracked position
    3. EMA smoothing: smooth_x = α·raw_x + (1-α)·prev
    4. Speed cap: limit movement per interval (camera inertia)
    5. Dead zone: skip keyframe if movement < threshold
    6. Velocity prediction: fill gaps when detection misses
    """
    logger.info(
        f"Detecting faces in {video_path.name} "
        f"from {start_time:.1f}s to {end_time:.1f}s (interval={interval}s)"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = min(1.0, _FACE_DOWNSCALE_HEIGHT / height) if height > 0 else 1.0
    down_w = int(width * scale)
    down_h = int(height * scale)

    # ── Set up detector (YuNet preferred, Haar fallback) ──────────
    yunet = None
    haar = None
    if _ensure_yunet_model():
        try:
            yunet = cv2.FaceDetectorYN.create(
                str(_YUNET_PATH), "", (down_w, down_h),
                score_threshold=0.5, nms_threshold=0.3, top_k=10,
            )
            logger.info(f"Face detector: YuNet DNN ({down_w}x{down_h})")
        except Exception as e:
            logger.warning(f"YuNet init failed ({e})")
    if yunet is None:
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        logger.info("Face detector: Haar cascade (fallback)")

    # ── Tracking state ────────────────────────────────────────────
    keyframes: list[FaceKeyframe] = []
    n_sampled = 0
    n_detected = 0

    smooth_x: float | None = None
    prev_raw_x: float | None = None
    velocity: float = 0.0
    miss_streak = 0
    last_emitted_x: float | None = None

    current_time = start_time

    while current_time <= end_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            current_time += interval
            continue

        n_sampled += 1
        t_rel = current_time - start_time

        # Downscale
        if scale < 1.0:
            small = cv2.resize(frame, (down_w, down_h), interpolation=cv2.INTER_AREA)
        else:
            small = frame

        faces = _detect_in_frame(small, yunet, haar)

        if len(faces) > 0:
            # ── Face detected ─────────────────────────────────────
            n_detected += 1
            raw_x = _pick_best_face(faces, smooth_x, scale)

            if prev_raw_x is not None:
                velocity = (raw_x - prev_raw_x) * _VELOCITY_DAMPING
            prev_raw_x = raw_x
            miss_streak = 0

            # EMA smoothing
            if smooth_x is None:
                smooth_x = raw_x
            else:
                target = _EMA_ALPHA * raw_x + (1 - _EMA_ALPHA) * smooth_x
                # Speed cap: limit how fast the camera can move per interval
                delta = target - smooth_x
                if abs(delta) > _MAX_SPEED_PX:
                    target = smooth_x + _MAX_SPEED_PX * (1 if delta > 0 else -1)
                smooth_x = target

        else:
            # ── No face — velocity prediction ─────────────────────
            miss_streak += 1
            if smooth_x is not None and miss_streak <= _MAX_MISS_STREAK:
                predicted = smooth_x + velocity
                # Speed cap on prediction too
                delta = predicted - smooth_x
                if abs(delta) > _MAX_SPEED_PX:
                    predicted = smooth_x + _MAX_SPEED_PX * (1 if delta > 0 else -1)
                smooth_x = predicted
                velocity *= 0.7
            elif smooth_x is None:
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

    # Pad boundaries for full clip coverage
    duration = end_time - start_time
    if keyframes and keyframes[0].time > 0.1:
        keyframes.insert(0, FaceKeyframe(time=0.0, x=keyframes[0].x))
    if keyframes and keyframes[-1].time < duration - 0.5:
        keyframes.append(FaceKeyframe(time=duration, x=keyframes[-1].x))

    det_rate = n_detected / max(n_sampled, 1) * 100
    det_name = "YuNet" if yunet else "Haar"
    logger.info(
        f"Face tracking ({det_name}): {n_detected}/{n_sampled} detections ({det_rate:.0f}%), "
        f"{len(keyframes)} keyframes (dead zone filtered)"
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
