import cv2
from pathlib import Path
from typing import List, Tuple
from video_maker.models import FaceKeyframe
from video_maker.utils import logger

_FACE_DOWNSCALE_HEIGHT = 480


def detect_faces(video_path: Path, start_time: float, end_time: float, interval: float = 3.0) -> List[FaceKeyframe]:
    """
    Detect the dominant face in the video segment and return its center X coordinate.
    Downscales frames to 480p for speed. Default interval=3s for fast scanning.
    """
    logger.info(f"Detecting faces in {video_path.name} from {start_time:.1f}s to {end_time:.1f}s (interval={interval}s)")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = min(1.0, _FACE_DOWNSCALE_HEIGHT / height) if height > 0 else 1.0
    
    keyframes = []
    current_time = start_time
    
    while current_time <= end_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Downscale for speed
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_x = width / 2
        
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            # Scale face center back to original coords
            center_x = (x + w / 2) / scale
            face_x = int((center_x / width) * 1920)
            
        keyframes.append(FaceKeyframe(time=current_time - start_time, x=face_x))
        current_time += interval
        
    cap.release()
    logger.info(f"Face detection complete: {len(keyframes)} keyframes")
    return keyframes

def get_optimal_crop(face_keyframes: List[FaceKeyframe], src_width: int, src_height: int) -> Tuple[int, int, int, int]:
    """
    Calculate the optimal 9:16 crop window based on face positions.
    Returns (crop_w, crop_h, crop_x, crop_y).
    """
    # Target aspect ratio: 9:16
    target_ratio = 9 / 16
    
    # Calculate crop dimensions based on source height (assuming height is always > width)
    crop_h = src_height
    crop_w = int(src_height * target_ratio)
    
    if crop_w > src_width:
        crop_w = src_width
        crop_h = int(src_width / target_ratio)
        
    # Make even
    crop_w -= (crop_w % 2)
    crop_h -= (crop_h % 2)
    
    # Average face X position (mapped from 1920 to real width)
    if face_keyframes:
        avg_face_x_1920 = sum(kf.x for kf in face_keyframes) / len(face_keyframes)
        avg_face_x = int(avg_face_x_1920 * src_width / 1920)
    else:
        avg_face_x = src_width // 2
        
    crop_x = int(max(0, min(src_width - crop_w, avg_face_x - (crop_w // 2))))
    crop_y = int((src_height - crop_h) // 2)
    
    return crop_w, crop_h, crop_x, crop_y
