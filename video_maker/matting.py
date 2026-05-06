"""Background removal using Robust Video Matting (RVM).

Pipeline:
    1. Lazy-load the RVM mobilenetv3 model via torch.hub (cached locally afterwards).
    2. Read the source clip frame by frame (OpenCV).
    3. Run RVM with a recurrent state for temporal coherence.
    4. Composite foreground on a black background (fg * alpha).
    5. Convert to grayscale (BT.601 luma).
    6. Pipe raw gray frames to FFmpeg → H.264 yuv420p output.

Designed to receive a clip already cropped + scaled (e.g. 1080x1920),
so RVM doesn't have to process the original 1080p landscape source.
"""

import subprocess
import sys as _sys
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from video_maker.config import FFMPEG_DIR
from video_maker.utils import logger

FFMPEG_BIN = str(FFMPEG_DIR / ("ffmpeg.exe" if _sys.platform == "win32" else "ffmpeg"))


# ── Model singleton (lazy-loaded, thread-safe) ──────────────────────

_model = None
_model_lock = threading.Lock()
_device = None


def _get_device():
    """Pick the best available torch device (cuda > mps > cpu)."""
    global _device
    if _device is not None:
        return _device
    import torch
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    logger.info(f"[matting] Using torch device: {_device}")
    return _device


def _load_model():
    """Lazy-load RVM (mobilenetv3) via torch.hub. Cached on disk after first call."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model

        import torch
        try:
            import torchvision  # noqa: F401 — RVM hub code imports this
        except ImportError as e:
            raise RuntimeError(
                "Le package 'torchvision' est requis pour le preset podcast (RVM). "
                "Installe-le dans ton venv : python -m pip install torchvision"
            ) from e

        device = _get_device()
        logger.info("[matting] Loading RVM mobilenetv3 (first call may download ~10 MB)...")

        try:
            model = torch.hub.load(
                "PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True
            )
        except Exception as e:
            logger.error(f"[matting] torch.hub.load failed: {e}")
            err = str(e).lower()
            if "torchvision" in err or "no module named" in err:
                raise RuntimeError(
                    "Impossible de charger RVM (dépendance manquante). "
                    "Exécute : python -m pip install torchvision"
                ) from e
            raise RuntimeError(
                "Échec du chargement du modèle RVM. "
                "Au premier lancement, une connexion Internet est nécessaire pour "
                "télécharger le dépôt (mis en cache dans ~/.cache/torch/hub)."
            ) from e

        model = model.eval().to(device)
        if device.type == "cuda":
            try:
                model = model.half()
                logger.info("[matting] Using FP16 inference on GPU")
            except Exception:
                pass

        _model = model
        logger.info("[matting] RVM model ready")
        return _model


def warmup() -> bool:
    """Optionally pre-load the model. Returns True if successful."""
    try:
        _load_model()
        return True
    except Exception as e:
        logger.warning(f"[matting] Warmup failed: {e}")
        return False


# ── Resolution helper ───────────────────────────────────────────────

def _pick_downsample_ratio(width: int, height: int) -> float:
    """Pick RVM's internal downsample_ratio based on input resolution.

    Reference values from the RVM authors:
      - 4K  → 0.125
      - 1080p → 0.25
      - 720p  → 0.4
      - SD    → 1.0
    """
    longest = max(width, height)
    if longest >= 2160:
        return 0.125
    if longest >= 1440:
        return 0.2
    if longest >= 1000:
        return 0.25
    if longest >= 700:
        return 0.4
    return 1.0


# ── Main entry point ────────────────────────────────────────────────

def matte_clip_to_bw_on_black(
    input_path: Path,
    output_path: Path,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Path:
    """Run RVM on a clip → composite fg on black → grayscale → write MP4.

    The output is a fully-rendered video clip (no alpha channel, ready for further
    FFmpeg composition like subtitle burn-in).

    Args:
        input_path: Source video (any size; resolution drives the downsample ratio).
        output_path: Output MP4 path (H.264 yuv420p).
        progress_cb: Optional callback receiving 0.0–1.0 progress.

    Returns:
        Path to output_path.
    """
    import torch

    model = _load_model()
    device = _get_device()
    use_half = device.type == "cuda"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for matting: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    downsample_ratio = _pick_downsample_ratio(src_w, src_h)
    logger.info(
        f"[matting] Input: {src_w}x{src_h} @ {src_fps:.2f}fps, "
        f"{total_frames} frames, downsample_ratio={downsample_ratio}"
    )

    # FFmpeg writer: receive raw gray frames on stdin → H.264 yuv420p on disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        FFMPEG_BIN, "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{src_w}x{src_h}",
        "-r", f"{src_fps:.4f}",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path).replace("\\", "/"),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    rec = [None] * 4  # RVM recurrent state (preserves temporal coherence)
    log_interval = max(1, total_frames // 10) if total_frames > 0 else 60
    frame_idx = 0

    try:
        with torch.no_grad():
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(frame_rgb).to(device)
                t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
                if use_half:
                    t = t.half()

                fgr, pha, *rec = model(t, *rec, downsample_ratio)

                composite = fgr * pha  # foreground on black, [1, 3, H, W] ∈ [0,1]

                # BT.601 luma → grayscale [1, H, W]
                gray = (
                    0.299 * composite[:, 0]
                    + 0.587 * composite[:, 1]
                    + 0.114 * composite[:, 2]
                )

                gray_np = (
                    gray.clamp(0, 1).squeeze(0).detach().cpu().float().numpy() * 255.0
                ).astype(np.uint8)

                if proc.stdin is None:
                    raise RuntimeError("FFmpeg writer stdin closed unexpectedly")
                proc.stdin.write(gray_np.tobytes())

                frame_idx += 1
                if total_frames > 0 and frame_idx % log_interval == 0:
                    pct = 100.0 * frame_idx / total_frames
                    logger.info(f"[matting] {frame_idx}/{total_frames} ({pct:.0f}%)")
                    if progress_cb:
                        try:
                            progress_cb(frame_idx / total_frames)
                        except Exception:
                            pass

        if proc.stdin is not None:
            proc.stdin.close()
        rc = proc.wait(timeout=120)
        if rc != 0:
            err = (proc.stderr.read() if proc.stderr else b"").decode(
                "utf-8", errors="replace"
            )[-2000:]
            raise RuntimeError(f"FFmpeg writer failed (rc={rc}):\n{err}")

        logger.info(f"[matting] Done: {output_path.name} ({frame_idx} frames)")
        if progress_cb:
            try:
                progress_cb(1.0)
            except Exception:
                pass
        return output_path

    finally:
        cap.release()
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
