import os
import time
import cv2
import streamlit as st
import pandas as pd

from core.logger import EventLogger
from core.config import AppConfig
from core.face_engine import create_face_app, faces_in_frame, normalize
from core.gallery import build_gallery
from core.yolo_engine import create_yolo, detect_people_food_drink
from core.trackers import create_tracker
from core.assoc import best_person_for_face, item_near_person

CFG = AppConfig()


def draw_box(frame, bbox, lines, color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    y = max(0, y1 - 10)
    for line in lines:
        cv2.putText(frame, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y -= 18


def load_log_tail(csv_path: str, n: int = 50) -> pd.DataFrame:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.DataFrame(columns=["detection_image", "violation_type", "station", "date", "time"])
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame(columns=["detection_image", "violation_type", "station", "date", "time"])
        return df.tail(n)
    except Exception:
        return pd.DataFrame(columns=["detection_image", "violation_type", "station", "date", "time"])


st.set_page_config(page_title="First Class Cabin Monitor", layout="wide")
st.title("First Class Cabin Monitor")

for p in [CFG.video_path, CFG.users_csv, CFG.faces_dir]:
    if not os.path.exists(p):
        st.error(f"Missing path: {p}")
        st.stop()

# Engines
face_app = create_face_app()
yolo = create_yolo(CFG.yolo_model)

# Gallery
gallery, gallery_mat, gallery_ids, gallery_names = build_gallery(face_app, CFG.users_csv, CFG.faces_dir)

# Logger
logger = EventLogger(
    csv_path=CFG.logs_csv,
    images_dir=CFG.detection_images_dir,
    station=CFG.station,
    cooldown_seconds=CFG.dedupe_cooldown_seconds,
)

tab_live, tab_users = st.tabs(["Live Monitor", "First Class Users"])

with tab_users:
    st.subheader("First Class Users")
    st.dataframe(
        [
            {
                "user_id": g["user_id"],
                "name": g["name"],
                "image": os.path.basename(g["image_path"]),
                "embedding_ok": g["embedding"] is not None,
                "error": g["error"],
            }
            for g in gallery
        ],
        use_container_width=True,
    )

with tab_live:
    col_feed, col_controls, col_status = st.columns([3, 1, 1])

    with col_feed:
        st.subheader("Live Feed")
        feed_ph = st.empty()

        st.subheader("Detections Log (latest)")
        log_table_ph = st.empty()

        # initial render
        log_table_ph.dataframe(load_log_tail(CFG.logs_csv, 50), use_container_width=True)

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

    if not st.session_state.running:
        status_ph.write("Stopped")
        st.stop()

    status_ph.write("Running")

    cap = cv2.VideoCapture(CFG.video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        st.stop()

    frame_i = 0
    trackers = []
    track_meta = []  # [{"identity","has_food","has_drink"}]

    # Throttle log table refresh so UI stays smooth
    LOG_REFRESH_EVERY_N_FRAMES = max(1, CFG.detect_interval * 2)

    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    continue

            frame_i += 1

            # resize
            h, w = frame.shape[:2]
            if w != CFG.resize_width:
                nh = int(h * CFG.resize_width / w)
                frame = cv2.resize(frame, (CFG.resize_width, nh), interpolation=cv2.INTER_AREA)

            # detect every N frames (re-init trackers + meta)
            if frame_i % CFG.detect_interval == 0 or not trackers:
                trackers.clear()
                track_meta.clear()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                persons, foods, drinks = detect_people_food_drink(yolo, rgb, CFG.yolo_conf)
                faces = faces_in_frame(face_app, frame)

                for f in faces:
                    face_box = tuple(map(int, f.bbox.tolist()))
                    person_box = best_person_for_face(face_box, persons) or face_box

                    # identity
                    identity = "no id"
                    emb = normalize(f.embedding)
                    if gallery_mat is not None:
                        sims = gallery_mat @ emb
                        idx = int(sims.argmax())
                        if float(sims[idx]) >= CFG.sim_threshold:
                            identity = gallery_ids[idx]

                    # items near person
                    has_food = any(
                        item_near_person(b, person_box, CFG.item_link_iou, CFG.item_link_dist)
                        for b in foods
                    )
                    has_drink = any(
                        item_near_person(b, person_box, CFG.item_link_iou, CFG.item_link_dist)
                        for b in drinks
                    )

                    # ---- REALTIME LOGGING (NO DUPLICATES: handled by EventLogger cooldown) ----
                    identity_key = identity if identity != "no id" else "unknown"
                    x1, y1, x2, y2 = person_box

                    if identity == "no id":
                        logger.log_violation(frame, (x1, y1, x2, y2), identity_key, "no_id")

                    if not has_food:
                        logger.log_violation(frame, (x1, y1, x2, y2), identity_key, "no_food")

                    if not has_drink:
                        logger.log_violation(frame, (x1, y1, x2, y2), identity_key, "no_drink")

                    # tracker init (track the person box)
                    tracker = create_tracker()
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

                    trackers.append(tracker)
                    track_meta.append(
                        {"identity": identity, "has_food": has_food, "has_drink": has_drink}
                    )

            # update trackers and draw
            for trk, meta in zip(trackers, track_meta):
                ok, box = trk.update(frame)
                if not ok:
                    continue
                x, y, bw, bh = map(int, box)
                bbox = (x, y, x + bw, y + bh)

                identity = meta["identity"]
                color = (0, 255, 0) if identity != "no id" else (0, 0, 255)

                draw_box(
                    frame,
                    bbox,
                    [
                        f"identity: {identity}",
                        f"has food: {'yes' if meta['has_food'] else 'no'}",
                        f"has drink: {'yes' if meta['has_drink'] else 'no'}",
                    ],
                    color,
                )

            # show frame
            feed_ph.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            stats_ph.write(f"Frame {frame_i} | Trackers {len(trackers)}")

            # refresh table occasionally
            if frame_i % LOG_REFRESH_EVERY_N_FRAMES == 0:
                log_table_ph.dataframe(load_log_tail(CFG.logs_csv, 50), use_container_width=True)

            time.sleep(CFG.frame_delay)

    finally:
        cap.release()
