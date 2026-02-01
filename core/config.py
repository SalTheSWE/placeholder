from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    video_path: str = "resources/input_video/IMG_3433.mp4"
    users_csv: str = "resources/csvs/firstclass_users.csv"
    faces_dir: str = "resources/faces"

    # perf
    resize_width: int = 640
    detect_interval: int = 15
    frame_delay: float = 0.01

    # recognition
    sim_threshold: float = 0.45

    # yolo
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    item_link_iou: float = 0.02
    item_link_dist: float = 160.0
    
    # logging
    logs_csv: str = "logs/detections.csv"
    detection_images_dir: str = "logs/detection_images"
    station: str = "first_class_cabin"
    dedupe_cooldown_seconds: int = 30
