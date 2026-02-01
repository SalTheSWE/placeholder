import os
import cv2
import numpy as np
import pandas as pd
from .face_engine import first_face_embedding

def load_users(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"user_id", "image", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df

def build_gallery(face_app, users_csv: str, faces_dir: str):
    df = load_users(users_csv)

    gallery = []
    for _, row in df.iterrows():
        img_path = os.path.join(faces_dir, str(row["image"]))
        img = cv2.imread(img_path)
        if img is None:
            gallery.append({
                "user_id": str(row["user_id"]),
                "name": str(row["name"]),
                "image_path": img_path,
                "embedding": None,
                "error": "image_not_found",
            })
            continue

        emb = first_face_embedding(face_app, img)
        if emb is None:
            gallery.append({
                "user_id": str(row["user_id"]),
                "name": str(row["name"]),
                "image_path": img_path,
                "embedding": None,
                "error": "no_face_detected",
            })
            continue

        gallery.append({
            "user_id": str(row["user_id"]),
            "name": str(row["name"]),
            "image_path": img_path,
            "embedding": emb.astype(np.float32),
            "error": None,
        })

    valid = [g for g in gallery if g["embedding"] is not None]
    if not valid:
        return gallery, None, None, None

    mat = np.stack([g["embedding"] for g in valid]).astype(np.float32)
    ids = [g["user_id"] for g in valid]
    names = [g["name"] for g in valid]
    return gallery, mat, ids, names
