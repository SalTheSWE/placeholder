def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def center_distance(a, b) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def best_person_for_face(face_box, person_boxes):
    if not person_boxes:
        return None
    best = max(person_boxes, key=lambda p: iou(face_box, p))
    if iou(face_box, best) > 0.001:
        return best
    # fallback to distance if IoU is tiny
    return min(person_boxes, key=lambda p: center_distance(face_box, p))

def item_near_person(item_box, person_box, iou_thr: float, dist_thr: float) -> bool:
    return (iou(item_box, person_box) >= iou_thr) or (center_distance(item_box, person_box) <= dist_thr)
