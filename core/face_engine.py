import numpy as np
import cv2
from insightface.app import FaceAnalysis

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-9)

def create_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_s")
    app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU fast
    return app

def faces_in_frame(face_app: FaceAnalysis, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return face_app.get(rgb)

def first_face_embedding(face_app: FaceAnalysis, img_bgr):
    if img_bgr is None:
        return None
    faces = faces_in_frame(face_app, img_bgr)
    if not faces:
        return None
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    return normalize(faces[0].embedding)
