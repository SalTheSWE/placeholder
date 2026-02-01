import numpy as np
from ultralytics import YOLO

# COCO ids for yolov8*.pt
COCO = {
    "person": 0,
    "bottle": 39,
    "wine_glass": 40,
    "cup": 41,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot_dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
}

DRINK_CLASSES = {COCO["bottle"], COCO["wine_glass"], COCO["cup"]}
FOOD_CLASSES = {
    COCO["bowl"], COCO["banana"], COCO["apple"], COCO["sandwich"], COCO["orange"],
    COCO["broccoli"], COCO["carrot"], COCO["hot_dog"], COCO["pizza"], COCO["donut"], COCO["cake"],
}

def create_yolo(model_name: str) -> YOLO:
    return YOLO(model_name)

def detect_people_food_drink(yolo: YOLO, frame_rgb, conf: float):
    out = yolo(frame_rgb, verbose=False)[0]
    persons, foods, drinks = [], [], []

    if out.boxes is None or len(out.boxes) == 0:
        return persons, foods, drinks

    boxes = out.boxes.xyxy.cpu().numpy()
    clses = out.boxes.cls.cpu().numpy().astype(int)
    confs = out.boxes.conf.cpu().numpy()

    for xyxy, cls, c in zip(boxes, clses, confs):
        if c < conf:
            continue
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        box = (x1, y1, x2, y2)
        if cls == COCO["person"]:
            persons.append(box)
        elif cls in FOOD_CLASSES:
            foods.append(box)
        elif cls in DRINK_CLASSES:
            drinks.append(box)

    return persons, foods, drinks
