from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LogEvent:
    detection_image: str
    violation_type: str
    station: str
    date: str
    time: str


class EventLogger:
    """
    Appends events to a CSV with de-duplication (cooldown window).
    Also saves a cropped image per logged event.
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        station: str,
        cooldown_seconds: int = 30,
    ):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.station = station
        self.cooldown_seconds = cooldown_seconds

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # last_logged[key] = epoch seconds
        self.last_logged: Dict[str, float] = {}
        self._ensure_csv_header()
        self._warm_start_from_csv(max_rows=500)

    def _ensure_csv_header(self) -> None:
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["detection_image", "violation_type", "station", "date", "time"])

    def _warm_start_from_csv(self, max_rows: int = 500) -> None:
        """
        Load recent rows so reruns don't duplicate events.
        We store a small memory of recently-logged keys.
        """
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return

        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return

        # only keep last N rows for warm-start
        rows = rows[-max_rows:]
        now = datetime.now().timestamp()

        for r in rows:
            vt = (r.get("violation_type") or "").strip()
            st = (r.get("station") or "").strip()
            # identity not in schema; we embed it in filename when saving images
            img = (r.get("detection_image") or "").strip()
            # key is (violation_type, station, image_basename_prefix)
            k = f"{vt}|{st}|{os.path.basename(img).split('__')[0]}"
            self.last_logged[k] = now  # treat as recent, to avoid spam on rerun

    def _now_fields(self) -> Tuple[str, str, float]:
        dt = datetime.now()
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"), dt.timestamp()

    def _make_key(self, violation_type: str, identity_key: str) -> str:
        return f"{violation_type}|{self.station}|{identity_key}"

    def _should_log(self, key: str, now_ts: float) -> bool:
        last = self.last_logged.get(key)
        if last is None:
            return True
        return (now_ts - last) >= self.cooldown_seconds

    def save_crop(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        identity_key: str,
    ) -> Optional[str]:
        """
        Saves a crop image. Returns relative/absolute path used in CSV.
        """
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        date_s, time_s, _ = self._now_fields()
        safe_time = time_s.replace(":", "-")
        # identity_key is embedded for dedupe tracing
        filename = f"{identity_key}__{date_s}__{safe_time}.jpg"
        path = os.path.join(self.images_dir, filename)

        ok = cv2.imwrite(path, crop)
        if not ok:
            return None
        return path

    def log_violation(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        identity_key: str,
        violation_type: str,
    ) -> Optional[LogEvent]:
        """
        If not a duplicate (cooldown), writes a row + saves crop image.
        Returns LogEvent if logged, else None.
        """
        date_s, time_s, now_ts = self._now_fields()
        key = self._make_key(violation_type, identity_key)

        if not self._should_log(key, now_ts):
            return None

        img_path = self.save_crop(frame_bgr, bbox_xyxy, identity_key)
        if img_path is None:
            return None

        event = LogEvent(
            detection_image=img_path,
            violation_type=violation_type,
            station=self.station,
            date=date_s,
            time=time_s,
        )

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([event.detection_image, event.violation_type, event.station, event.date, event.time])

        self.last_logged[key] = now_ts
        return event
