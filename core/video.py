import cv2

def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap

def loop_read(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    if ok:
        return frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    return frame if ok else None

def resize_keep_aspect(frame_bgr, target_w: int):
    h, w = frame_bgr.shape[:2]
    if w == target_w:
        return frame_bgr
    nh = int(h * target_w / w)
    return cv2.resize(frame_bgr, (target_w, nh), interpolation=cv2.INTER_AREA)
