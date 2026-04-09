"""Local scoring engine: audio (librosa) + visual (OpenCV) analysis without any LLM."""

import subprocess
import time
import numpy as np
import librosa
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from video_maker.config import (
    FFMPEG_DIR,
    SCORING_WINDOW,
    SCORING_HOP,
    TOP_PRESCORE,
    CLIP_MIN_DURATION,
    CLIP_MAX_DURATION,
    NUM_WORKERS,
)
from video_maker.models import ScoredSegment
from video_maker.utils import logger

import sys as _sys
FFMPEG_BIN = str(FFMPEG_DIR / ("ffmpeg.exe" if _sys.platform == "win32" else "ffmpeg"))
FFPROBE_BIN = str(FFMPEG_DIR / ("ffprobe.exe" if _sys.platform == "win32" else "ffprobe"))

_SAMPLE_RATE = 22050  # librosa default
_CHUNK_OVERLAP = 2.0  # seconds of overlap between audio chunks for clean stitching


# ── Audio extraction ────────────────────────────────────────────────


def _extract_audio(video_path: Path, out_path: Path) -> Path:
    """Extract mono WAV audio from video using FFmpeg."""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(_SAMPLE_RATE),
        "-ac", "1",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[-500:]}")
    logger.info(f"Audio extracted: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_path


# ── Audio analysis ──────────────────────────────────────────────────


def _analyze_audio_array(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute audio features from a numpy array (works on full or chunk)."""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    S = np.abs(librosa.stft(y, hop_length=512))
    spectral_flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))
    spectral_flux = np.pad(spectral_flux, (0, max(0, len(rms) - len(spectral_flux))))
    spectral_flux = spectral_flux[: len(rms)]
    return rms, onset_env[: len(rms)], spectral_flux


def _extract_audio_chunk(
    video_path_str: str, start: float, duration: float, out_path_str: str
) -> str:
    """Extract a time-slice of audio from video. Thread-safe (no shared state)."""
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path_str,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(_SAMPLE_RATE), "-ac", "1",
        out_path_str,
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    return out_path_str


def _analyze_audio_chunk(chunk_path_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load and analyze a single audio chunk. Top-level function for ProcessPoolExecutor."""
    y, sr = librosa.load(chunk_path_str, sr=_SAMPLE_RATE, mono=True)
    dur = librosa.get_duration(y=y, sr=sr)
    rms, onset, flux = _analyze_audio_array(y, sr)
    return rms, onset, flux, dur


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def _analyze_audio(audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Single-file fallback: compute per-frame audio features using librosa."""
    y, sr = librosa.load(str(audio_path), sr=_SAMPLE_RATE, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    logger.info(f"Audio loaded: {duration:.1f}s, {sr} Hz")
    rms, onset_env, spectral_flux = _analyze_audio_array(y, sr)
    return rms, onset_env, spectral_flux, duration


def _frames_to_time(frame_indices: np.ndarray) -> np.ndarray:
    """Convert librosa frame indices to seconds."""
    return librosa.frames_to_time(frame_indices, sr=_SAMPLE_RATE, hop_length=512)


def _score_audio_windows(
    rms: np.ndarray,
    onset_env: np.ndarray,
    spectral_flux: np.ndarray,
    duration: float,
) -> List[Tuple[float, float, float]]:
    """Score overlapping windows using audio features.

    Returns list of (start_sec, end_sec, audio_score) sorted by score desc.
    """
    frames_per_sec = _SAMPLE_RATE / 512
    window_frames = int(SCORING_WINDOW * frames_per_sec)
    hop_frames = int(SCORING_HOP * frames_per_sec)
    total_frames = len(rms)

    windows = []
    pos = 0
    while pos + window_frames <= total_frames:
        r = rms[pos : pos + window_frames]
        o = onset_env[pos : pos + window_frames]
        sf = spectral_flux[pos : pos + window_frames]

        # Combine: mean energy + peak energy + onset intensity + spectral variation
        score = (
            0.3 * float(np.mean(r))
            + 0.2 * float(np.max(r))
            + 0.3 * float(np.mean(o))
            + 0.2 * float(np.mean(sf))
        )

        start_sec = pos / frames_per_sec
        end_sec = start_sec + SCORING_WINDOW
        windows.append((start_sec, min(end_sec, duration), score))
        pos += hop_frames

    return windows


# ── Visual analysis ─────────────────────────────────────────────────


_VISUAL_DOWNSCALE_HEIGHT = 480  # downscale frames for faster face detection
_VISUAL_TOP_N = 30  # only run visual analysis on top N audio windows
_SAMPLES_PER_WINDOW = 3  # sample start / middle / end of each window


def _score_one_window(
    video_path_str: str,
    win_idx: int,
    start_sec: float,
    end_sec: float,
    scale: float,
    scaled_area: float,
) -> Tuple[int, float]:
    """Score a single window visually. Opens its own VideoCapture (thread-safe)."""
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        return (win_idx, 0.0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    dur = end_sec - start_sec
    sample_times = [start_sec, start_sec + dur / 2, end_sec - 0.5]

    motion_scores = []
    face_ratios = []
    face_sizes = []
    prev_gray = None

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_scores.append(float(np.mean(diff)) / 255.0)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            face_ratios.append(1.0)
            largest = max(faces, key=lambda f: f[2] * f[3])
            face_sizes.append((largest[2] * largest[3]) / scaled_area)
        else:
            face_ratios.append(0.0)
            face_sizes.append(0.0)

        prev_gray = gray

    cap.release()

    avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.0
    face_presence = float(np.mean(face_ratios)) if face_ratios else 0.0
    avg_face_size = float(np.mean(face_sizes)) if face_sizes else 0.0

    return (win_idx, 0.3 * avg_motion + 0.4 * face_presence + 0.3 * avg_face_size)


def _analyze_visual_windows(
    video_path: Path,
    windows: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    """Add visual scores to the top audio-scored windows using parallel workers.

    Each worker opens its own VideoCapture for thread safety.
    Only analyses the top _VISUAL_TOP_N windows by audio score.
    Samples _SAMPLES_PER_WINDOW frames per window (start/mid/end).

    Returns list of (start, end, audio_score, visual_score) for ALL windows
    (non-analysed windows get visual_score=0).
    """
    # Pre-filter: only visually analyse the best audio windows
    indexed_sorted = sorted(
        enumerate(windows), key=lambda x: x[1][2], reverse=True
    )
    top_indices = set(i for i, _ in indexed_sorted[:_VISUAL_TOP_N])
    n_analyse = min(len(windows), _VISUAL_TOP_N)

    logger.info(
        f"Visual analysis on {n_analyse}/{len(windows)} windows, "
        f"{_SAMPLES_PER_WINDOW} frames each, {NUM_WORKERS} workers "
        f"(downscaled to {_VISUAL_DOWNSCALE_HEIGHT}p)"
    )

    # Get video dimensions for scale computation
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video for visual analysis: {video_path}")
        return [(s, e, a, 0.0) for s, e, a in windows]
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    scale = min(1.0, _VISUAL_DOWNSCALE_HEIGHT / vid_height) if vid_height > 0 else 1.0
    scaled_area = max((vid_width * scale) * (vid_height * scale), 1)

    # Build work items
    work = [(i, s, e) for i, (s, e, _) in enumerate(windows) if i in top_indices]
    video_str = str(video_path)

    visual_map: dict[int, float] = {}
    done_count = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {
            pool.submit(_score_one_window, video_str, i, s, e, scale, scaled_area): i
            for i, s, e in work
        }
        for future in as_completed(futures):
            win_idx, score = future.result()
            visual_map[win_idx] = score
            done_count += 1
            if done_count % 5 == 0 or done_count == n_analyse:
                logger.info(f"Visual analysis progress: {done_count}/{n_analyse}")

    logger.info(f"Visual analysis complete: {done_count} windows processed")

    # Rebuild full results list
    results: List[Tuple[float, float, float, float]] = []
    for idx, (s, e, a) in enumerate(windows):
        results.append((s, e, a, visual_map.get(idx, 0.0)))
    return results


# ── Normalization & ranking ─────────────────────────────────────────


def _normalize(values: List[float]) -> List[float]:
    """Min-max normalize a list of floats to 0-1."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _merge_overlapping(segments: List[ScoredSegment], max_gap: float = 15.0) -> List[ScoredSegment]:
    """Merge overlapping or nearly-adjacent segments, keeping the best score."""
    if not segments:
        return segments

    sorted_segs = sorted(segments, key=lambda s: s.start)
    merged = [sorted_segs[0].model_copy()]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        if seg.start <= last.end + max_gap:
            # Extend the merged segment
            last.end = max(last.end, seg.end)
            last.audio_score = max(last.audio_score, seg.audio_score)
            last.visual_score = max(last.visual_score, seg.visual_score)
            last.total_score = max(last.total_score, seg.total_score)
        else:
            merged.append(seg.model_copy())

    return merged


def _expand_to_min_duration(
    segments: List[ScoredSegment],
    video_duration: float,
) -> List[ScoredSegment]:
    """Expand segments to meet CLIP_MIN_DURATION, avoiding overlaps."""
    expanded = []

    for seg in segments:
        current_dur = seg.end - seg.start
        if current_dur >= CLIP_MIN_DURATION:
            expanded.append(seg)
            continue

        deficit = CLIP_MIN_DURATION - current_dur
        half = deficit / 2.0

        new_start = max(0.0, seg.start - half)
        new_end = min(video_duration, seg.end + half)

        # If one side is clamped, give the remainder to the other side
        if new_start == 0.0:
            new_end = min(video_duration, new_start + CLIP_MIN_DURATION)
        if new_end == video_duration:
            new_start = max(0.0, new_end - CLIP_MIN_DURATION)

        # Cap at CLIP_MAX_DURATION
        if new_end - new_start > CLIP_MAX_DURATION:
            new_end = new_start + CLIP_MAX_DURATION

        seg_copy = seg.model_copy()
        seg_copy.start = round(new_start, 2)
        seg_copy.end = round(new_end, 2)
        expanded.append(seg_copy)

    # Strict: only keep segments that actually reach min duration
    before_filter = len(expanded)
    expanded = [s for s in expanded if (s.end - s.start) >= CLIP_MIN_DURATION - 0.5]
    logger.info(
        f"Expand: {before_filter} segments → {len(expanded)} after min-duration filter "
        f"(≥{CLIP_MIN_DURATION}s)"
    )

    # Remove overlaps: greedily keep highest-scored non-overlapping segments
    expanded.sort(key=lambda s: s.total_score, reverse=True)
    kept = []
    for seg in expanded:
        overlaps = any(
            not (seg.end <= k.start or seg.start >= k.end)
            for k in kept
        )
        if not overlaps:
            kept.append(seg)

    logger.info(f"Overlap removal: {len(expanded)} → {len(kept)} non-overlapping segments")
    kept.sort(key=lambda s: s.start)
    return kept


# ── Public API ──────────────────────────────────────────────────────


def score_video(video_path: Path, work_dir: Path) -> Tuple[List[ScoredSegment], Path, float]:
    """Run full local scoring on a video using parallel workers.

    Pipeline:
      1. Probe duration
      2. Extract FULL audio for both scoring and Whisper (one pass)
      3. Analyse audio in parallel chunks (ProcessPool)
      4. Stitch features, score windows
      5. Visual analysis on top windows (ThreadPool — already parallel)

    Returns:
        segments: Top pre-scored segments (up to TOP_PRESCORE), sorted by score desc
        audio_path: Path to extracted WAV (for Whisper later)
        duration: Video duration in seconds
    """
    logger.info(f"Starting local scoring of {video_path.name}")
    t0 = time.time()

    # Step 1: Probe total duration
    duration = _get_video_duration(video_path)
    logger.info(f"Video duration: {duration:.1f}s")

    # Step 2: Extract full audio for everyone
    t_audio = time.time()
    audio_path = work_dir / "audio_full.wav"
    _extract_audio(video_path, audio_path)
    logger.info(f"[TIMING] Full audio extraction: {time.time() - t_audio:.1f}s")

    # Decide chunk count: ~2 min per chunk, at least 1, at most NUM_WORKERS
    n_chunks = min(NUM_WORKERS, max(1, int(duration / 120)))
    chunk_dur = duration / n_chunks
    overlap = _CHUNK_OVERLAP
    frames_per_sec = _SAMPLE_RATE / 512

    # Step 3: Extract audio chunks from the full WAV in parallel (very fast seek in WAV)
    t1 = time.time()
    logger.info(f"Extracting {n_chunks} parallel audio chunks (~{chunk_dur:.0f}s each) from WAV...")

    chunk_meta: list[Tuple[int, float, str]] = []  # (idx, real_start, path)
    extract_args = []
    audio_path_str = str(audio_path)
    for i in range(n_chunks):
        c_start = max(0.0, i * chunk_dur - (overlap if i > 0 else 0))
        c_end = min(duration, (i + 1) * chunk_dur + (overlap if i < n_chunks - 1 else 0))
        c_dur = c_end - c_start
        out_p = str(work_dir / f"audio_chunk_{i:02d}.wav")
        # Use audio_path (WAV) as input for faster extraction than from video
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{c_start:.3f}",
            "-t", f"{c_dur:.3f}",
            "-i", audio_path_str,
            "-acodec", "copy", # Just copy since it's already PCM
            out_p,
        ]
        extract_args.append(cmd)
        chunk_meta.append((i, c_start, out_p))

    def _run_cmd(cmd):
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        list(pool.map(_run_cmd, extract_args))

    logger.info(f"[TIMING] Parallel audio extraction from WAV: {time.time() - t1:.1f}s")

    # Step 4: Analyse each chunk with librosa in parallel (CPU-bound → ProcessPool)
    t2 = time.time()
    logger.info(f"Analyzing {n_chunks} audio chunks with librosa (ProcessPool)...")

    chunk_results: dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    with ProcessPoolExecutor(max_workers=n_chunks) as pool:
        futs_map = {
            pool.submit(_analyze_audio_chunk, path): idx
            for idx, _, path in chunk_meta
        }
        for f in as_completed(futs_map):
            idx = futs_map[f]
            chunk_results[idx] = f.result()
            logger.info(f"  Chunk {idx + 1}/{n_chunks} analysed")

    logger.info(f"[TIMING] Parallel librosa analysis ({n_chunks} chunks): {time.time() - t2:.1f}s")

    # Step 5: Stitch feature arrays
    all_rms, all_onset, all_flux = [], [], []
    for i in range(n_chunks):
        rms_c, onset_c, flux_c, _ = chunk_results[i]
        if i > 0:
            trim_frames = int(overlap * frames_per_sec)
            rms_c = rms_c[trim_frames:]
            onset_c = onset_c[trim_frames:]
            flux_c = flux_c[trim_frames:]
        all_rms.append(rms_c)
        all_onset.append(onset_c)
        all_flux.append(flux_c)

    rms = np.concatenate(all_rms)
    onset_env = np.concatenate(all_onset)
    spectral_flux = np.concatenate(all_flux)

    # Cleanup chunk files
    for _, _, path in chunk_meta:
        try:
            Path(path).unlink()
        except OSError:
            pass

    # Step 6: Score audio windows
    logger.info("Scoring audio windows...")
    audio_windows = _score_audio_windows(rms, onset_env, spectral_flux, duration)
    logger.info(f"Generated {len(audio_windows)} audio windows")

    # Step 7: Visual analysis on top audio windows (already parallel)
    t3 = time.time()
    logger.info("Analyzing visual features (OpenCV) on top audio windows...")
    scored_windows = _analyze_visual_windows(video_path, audio_windows)
    logger.info(f"[TIMING] Visual analysis: {time.time() - t3:.1f}s")

    # Step 8: Normalize and combine
    audio_scores = [w[2] for w in scored_windows]
    visual_scores = [w[3] for w in scored_windows]
    norm_audio = _normalize(audio_scores)
    norm_visual = _normalize(visual_scores)

    segments = []
    for i, (start, end, _, _) in enumerate(scored_windows):
        total = 0.55 * norm_audio[i] + 0.45 * norm_visual[i]
        segments.append(ScoredSegment(
            start=round(start, 2),
            end=round(end, 2),
            audio_score=round(norm_audio[i], 4),
            visual_score=round(norm_visual[i], 4),
            total_score=round(total, 4),
        ))

    # Step 9: Sort by total score, take top N
    segments.sort(key=lambda s: s.total_score, reverse=True)
    top = segments[:TOP_PRESCORE]

    logger.info(
        f"Top {len(top)} segments selected (scores: "
        f"{top[0].total_score:.3f} ... {top[-1].total_score:.3f})"
    )

    logger.info(f"[TIMING] Total scoring: {time.time() - t0:.1f}s")
    return top, audio_path, duration


def rescore_with_text(segments: List[ScoredSegment]) -> List[ScoredSegment]:
    """Re-rank segments after Whisper has populated text_score.

    Final formula: 0.4 * audio + 0.35 * visual + 0.25 * text_richness
    """
    for seg in segments:
        seg.total_score = round(
            0.40 * seg.audio_score
            + 0.35 * seg.visual_score
            + 0.25 * seg.text_score,
            4,
        )

    segments.sort(key=lambda s: s.total_score, reverse=True)
    return segments
