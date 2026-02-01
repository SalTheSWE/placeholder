import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from insightface.app import FaceAnalysis

# -----------------------------
# Paths
# -----------------------------
VIDEO_PATH = "resources/input_video/IMG_3433.mp4"
USERS_CSV = "resources/csvs/firstclass_users.csv"
FACES_DIR = "resources/faces"

# -----------------------------
# Fixed parameters
# -----------------------------
RESIZE_WIDTH = 640
SIM_THRESHOLD = 0.45
FRAME_DELAY = 0.01
DETECT_INTERVAL = 15

# -----------------------------
# Face model
# -----------------------------
@st.cache_resource
def get_face_app():
    app = FaceAnalysis(name="buffalo_s")
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app


# -----------------------------
# Embedding helpers
# -----------------------------
def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-9)


def first_face_embedding(face_app, img_bgr):
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img_rgb)
    if not faces:
        return None

    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True,
    )

    return normalize(faces[0].embedding)


# -----------------------------
# Gallery builder
# -----------------------------
@st.cache_data
def build_gallery(csv_path, faces_dir):
    df = pd.read_csv(csv_path)

    required_cols = {"user_id", "image", "name"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain: user_id,image,name")

    face_app = get_face_app()

    gallery = []

    for _, row in df.iterrows():
        path = os.path.join(faces_dir, row["image"])
        img = cv2.imread(path)

        if img is None:
            print("Image missing:", path)
            continue

        emb = first_face_embedding(face_app, img)

        if emb is None:
            print("No face detected:", path)
            continue

        gallery.append(
            {
                "user_id": str(row["user_id"]),
                "name": str(row["name"]),
                "image_path": path,
                "embedding": emb,
            }
        )

    if not gallery:
        return [], None, None

    mat = np.stack([g["embedding"] for g in gallery]).astype(np.float32)
    ids = [g["user_id"] for g in gallery]

    return gallery, mat, ids


# -----------------------------
# Drawing helper
# -----------------------------
def draw_box(frame, bbox, text, color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        text,
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="First Class Cabin Monitor", layout="wide")
st.title("First Class Cabin Monitor")

for p in [VIDEO_PATH, USERS_CSV, FACES_DIR]:
    if not os.path.exists(p):
        st.error(f"Missing path: {p}")
        st.stop()

gallery, gallery_mat, gallery_ids = build_gallery(USERS_CSV, FACES_DIR)

tab_live, tab_users = st.tabs(["Live Monitor", "First Class Users"])

# -----------------------------
# Users tab
# -----------------------------
with tab_users:
    st.subheader("First Class Users")

    rows = [
        {
            "user_id": g["user_id"],
            "name": g["name"],
            "embedding_loaded": g["embedding"] is not None,
        }
        for g in gallery
    ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if gallery:
        options = [f'{g["user_id"]} - {g["name"]}' for g in gallery]
        sel = st.selectbox("Select passenger", options)

        if st.button("Show Image"):
            uid = sel.split(" - ")[0]
            entry = next(x for x in gallery if x["user_id"] == uid)
            st.image(entry["image_path"], width=300)


# -----------------------------
# Live Monitor
# -----------------------------
with tab_live:
    col_feed, col_controls, col_status = st.columns([3, 1, 1])

    with col_feed:
        st.subheader("Live Feed")
        feed_ph = st.empty()

    with col_controls:
        st.subheader("Controls")
        start = st.button("Start live feed")
        stop = st.button("Stop live feed")

    with col_status:
        status_ph = st.empty()
        stats_ph = st.empty()

    if "running" not in st.session_state:
        st.session_state.running = False

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

    if st.session_state.running:
        face_app = get_face_app()
        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            st.error("Could not open video.")
            st.stop()

        status_ph.write("Running")

        frame_i = 0
        trackers = []
        tracker_ids = []

        try:
            while st.session_state.running:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_i += 1

                h, w = frame.shape[:2]
                if w != RESIZE_WIDTH:
                    nh = int(h * RESIZE_WIDTH / w)
                    frame = cv2.resize(frame, (RESIZE_WIDTH, nh))

                matches = 0

                # Detection step
                if frame_i % DETECT_INTERVAL == 0 or not trackers:
                    trackers.clear()
                    tracker_ids.clear()

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = face_app.get(rgb)

                    for f in faces:
                        emb = normalize(f.embedding)

                        if gallery_mat is not None:
                            sims = gallery_mat @ emb
                            idx = int(np.argmax(sims))
                            sim = float(sims[idx])
                        else:
                            idx, sim = -1, -1

                        identity = (
                            gallery_ids[idx]
                            if idx >= 0 and sim >= SIM_THRESHOLD
                            else "no id"
                        )

                        if identity != "no id":
                            matches += 1

                        x1, y1, x2, y2 = map(int, f.bbox)
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

                        trackers.append(tracker)
                        tracker_ids.append(identity)

                # Tracking step
                for tracker, identity in zip(trackers, tracker_ids):
                    ok, box = tracker.update(frame)
                    if not ok:
                        continue

                    x, y, wbox, hbox = map(int, box)
                    color = (0, 255, 0) if identity != "no id" else (0, 0, 255)

                    draw_box(
                        frame,
                        (x, y, x + wbox, y + hbox),
                        f"identity: {identity}",
                        color,
                    )

                feed_ph.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

                stats_ph.write(
                    f"Frame {frame_i} | Trackers {len(trackers)} | Matches {matches}"
                )

                time.sleep(FRAME_DELAY)

        finally:
            cap.release()

    else:
        status_ph.write("Stopped")
