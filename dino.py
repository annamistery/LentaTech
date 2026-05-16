import os
import cv2
import json
import random
import shutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# =========================================================
# CONFIG
# =========================================================

MODEL_DIR = r"grounding-dino-base"
VIDEO_DIR = r"video_price"  # Лента #video_price
WORK_DIR = r"work_price_tags"


TEXT_PROMPT = "shelf price tag. price label. retail price tag. barcode label."


BOX_THRESHOLD = 0.28
TEXT_THRESHOLD = 0.2


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v")
FRAME_STEP = 10


MIN_BRIGHTNESS = 35
MIN_LAPLACIAN_VAR = 60


VAL_RATIO = 0.2
RANDOM_SEED = 42


CLASS_ID = 0
CLASS_NAME = "price_tag"


ROTATE_180_BEFORE_FLIP = False
HORIZONTAL_FLIP = True


SAVE_VIS = True
SAVE_EMPTY_FRAMES = False


# -----------------------------
# ЖЁСТКИЕ ОГРАНИЧЕНИЯ ДЛЯ ЦЕННИКОВ
# -----------------------------
MIN_BOX_WIDTH_PX = 20
MIN_BOX_HEIGHT_PX = 12


MIN_BOX_WIDTH_NORM = 0.015
MIN_BOX_HEIGHT_NORM = 0.012


MAX_BOX_WIDTH_NORM = 0.28
MAX_BOX_HEIGHT_NORM = 0.22


# Площадь бокса как доля площади кадра
MIN_BOX_AREA_NORM = 0.0002
MAX_BOX_AREA_NORM = 0.06


# Соотношение сторон бокса
MIN_ASPECT_RATIO = 0.6    # слишком узкие вертикальные полосы отсекаем
MAX_ASPECT_RATIO = 6.5    # слишком длинные горизонтальные рамки отсекаем


# Если бокс почти полностью внутри другого, удаляем меньший/худший
CONTAINMENT_THRESHOLD = 0.90


# NMS для дублей
NMS_IOU_THRESHOLD = 0.35


# Доп. фильтр для «рамка слишком большая для ценника»
MAX_EDGE_SHARE = 0.32  # ни одна сторона не должна занимать > 32% кадра


# --- Размеры в пикселях (для кадров 720×1280) ---
# MIN_BOX_WIDTH_PX  = 25
# MIN_BOX_HEIGHT_PX = 15


# --- Нормализованные минимумы (согласованы с пиксельными) ---
# MIN_BOX_WIDTH_NORM  = 0.030   # 25 / 720  ≈ 21px min
# MIN_BOX_HEIGHT_NORM = 0.012   # 15 / 1280 ≈ 15px min


# --- Нормализованные максимумы ---
# W: ценник не шире ~210px = 29% кадра
# H: ценник не выше ~160px = 12.5% кадра (!) — ключевое изменение
# MAX_BOX_WIDTH_NORM  = 0.30   # объединено с MAX_EDGE_SHARE_W
# MAX_BOX_HEIGHT_NORM = 0.13    # было 0.22 → 282px, теперь 0.13 → 166px


# --- Площадь ---
# MIN_BOX_AREA_NORM = 0.00008   # ~74px² — поймать маленький штрихкод
# MAX_BOX_AREA_NORM = 0.035     # ~32 000px² ≈ 200×160px — было 55 000px²


# --- Соотношение сторон ---
# MIN_ASPECT_RATIO = 0.45   # чуть мягче для вертикальных shelf-label
# MAX_ASPECT_RATIO = 7.0    # немного шире для длинных горизонтальных ценников


# --- NMS ---
# NMS_IOU_THRESHOLD = 0.5  # без изменений


# --- Containment ---
# CONTAINMENT_THRESHOLD = 0.90  # без изменений


# --- Граничный фильтр: РАЗДЕЛИТЬ на две оси ---
# MAX_EDGE_SHARE_W = 0.30   # ни одна сторона не занимает > 30% ширины кадра
# MAX_EDGE_SHARE_H = 0.13   # ни одна сторона не занимает > 13% высоты кадра
# =========================================================
# PATHS
# =========================================================


WORK_DIR = Path(WORK_DIR)
YOLO_ROOT = WORK_DIR / "dataset"


IMAGES_ALL_DIR = YOLO_ROOT / "images_all"
LABELS_ALL_DIR = YOLO_ROOT / "labels_all"


IMAGES_TRAIN_DIR = YOLO_ROOT / "images" / "train"
IMAGES_VAL_DIR = YOLO_ROOT / "images" / "val"
LABELS_TRAIN_DIR = YOLO_ROOT / "labels" / "train"
LABELS_VAL_DIR = YOLO_ROOT / "labels" / "val"


VIS_DIR = YOLO_ROOT / "vis"
META_DIR = YOLO_ROOT / "meta"


for p in [
    WORK_DIR,
    YOLO_ROOT,
    IMAGES_ALL_DIR,
    LABELS_ALL_DIR,
    IMAGES_TRAIN_DIR,
    IMAGES_VAL_DIR,
    LABELS_TRAIN_DIR,
    LABELS_VAL_DIR,
    VIS_DIR,
    META_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD LOCAL MODEL ONLY
# =========================================================


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Loading local model from:", MODEL_DIR)


processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)


model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_DIR,
    local_files_only=True
).to(device)


model.eval()
print("Local Grounding DINO loaded successfully.")


# =========================================================
# HELPERS
# =========================================================


def is_good_frame(frame_bgr, min_brightness=35, min_laplacian_var=60):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_good = brightness >= min_brightness and sharpness >= min_laplacian_var
    return is_good, brightness, sharpness


def preprocess_frame(frame_bgr):
    if ROTATE_180_BEFORE_FLIP:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if HORIZONTAL_FLIP:
        frame_bgr = cv2.flip(frame_bgr, 1)
    return frame_bgr


def detect_price_tags(pil_image):
    inputs = processor(
        images=pil_image,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(pil_image.height, pil_image.width)]
    )[0]

    return results


def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return xc, yc, bw, bh


def clip_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), img_w - 1))
    y1 = max(0.0, min(float(y1), img_h - 1))
    x2 = max(0.0, min(float(x2), img_w - 1))
    y2 = max(0.0, min(float(y2), img_h - 1))
    return [x1, y1, x2, y2]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def containment_ratio(inner, outer):
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a = box_area(inner)
    return inter / a if a > 0 else 0.0


def valid_box(box, img_w, img_h):
    x1, y1, x2, y2 = clip_box(box, img_w, img_h)
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False, "non_positive"

    if w < MIN_BOX_WIDTH_PX or h < MIN_BOX_HEIGHT_PX:
        return False, "too_small_px"

    wn = w / img_w
    hn = h / img_h
    area_n = (w * h) / (img_w * img_h)
    aspect = w / h

    if wn < MIN_BOX_WIDTH_NORM or hn < MIN_BOX_HEIGHT_NORM:
        return False, "too_small_norm"

    if wn > MAX_BOX_WIDTH_NORM or hn > MAX_BOX_HEIGHT_NORM:
        return False, "too_large_norm"

    if area_n < MIN_BOX_AREA_NORM:
        return False, "too_small_area"

    if area_n > MAX_BOX_AREA_NORM:
        return False, "too_large_area"

    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False, "bad_aspect"

    if max(wn, hn) > MAX_EDGE_SHARE:
        return False, "too_large_edge"

    # if wn > MAX_EDGE_SHARE_W or hn > MAX_EDGE_SHARE_H:
        # return False, "too_large_edge"

    return True, "ok"


def nms_boxes(boxes, scores, iou_threshold=0.35):
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores_arr.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(
            inter), where=union > 0)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def remove_contained_boxes(boxes, scores, labels, containment_threshold=0.90):
    if len(boxes) <= 1:
        return boxes, scores, labels, 0

    keep = [True] * len(boxes)
    removed = 0

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue

            ci = containment_ratio(boxes[i], boxes[j])
            cj = containment_ratio(boxes[j], boxes[i])

            # Если box i почти внутри j, оставляем лучший по score
            if ci >= containment_threshold:
                if scores[i] <= scores[j]:
                    keep[i] = False
                    removed += 1
                    break

            # Если j почти внутри i и j хуже, удаляем j
            if cj >= containment_threshold and scores[j] < scores[i]:
                keep[j] = False
                removed += 1

    new_boxes = [b for b, k in zip(boxes, keep) if k]
    new_scores = [s for s, k in zip(scores, keep) if k]
    new_labels = [l for l, k in zip(labels, keep) if k]

    return new_boxes, new_scores, new_labels, removed


def save_visualization(frame_bgr, boxes, scores, labels, out_path):
    vis = frame_bgr.copy()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        text = f"{label} {float(score):.2f}"
        cv2.putText(
            vis,
            text,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imwrite(str(out_path), vis)


# =========================================================
# FIND VIDEOS
# =========================================================

video_files = [
    f for f in sorted(os.listdir(VIDEO_DIR))
    if f.lower().endswith(VIDEO_EXTS)
]


print(f"Found videos: {len(video_files)}")
for vf in video_files:
    print(" -", vf)


if len(video_files) == 0:
    raise FileNotFoundError(f"No video files found in: {VIDEO_DIR}")


# =========================================================
# PROCESS VIDEOS
# =========================================================


stats = {
    "videos_total": len(video_files),
    "frames_seen": 0,
    "frames_sampled": 0,
    "frames_good": 0,
    "frames_labeled": 0,
    "frames_empty_saved": 0,
    "frames_skipped_quality": 0,
    "frames_skipped_no_boxes": 0,
    "boxes_filtered_invalid": 0,
    "boxes_filtered_nms": 0,
    "boxes_filtered_contained": 0,
    "reject_reasons": {},
    "video_stats": {}
}


for video_name in video_files:
    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_name}")
        continue

    frame_idx = 0
    saved_from_video = 0
    skipped_quality = 0
    skipped_no_boxes = 0

    video_stem = Path(video_name).stem
    print(f"\\nProcessing: {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stats["frames_seen"] += 1

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        stats["frames_sampled"] += 1
        frame = preprocess_frame(frame)

        good, brightness, sharpness = is_good_frame(
            frame,
            min_brightness=MIN_BRIGHTNESS,
            min_laplacian_var=MIN_LAPLACIAN_VAR
        )

        if not good:
            stats["frames_skipped_quality"] += 1
            skipped_quality += 1
            frame_idx += 1
            continue

        stats["frames_good"] += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        try:
            results = detect_price_tags(pil_image)
        except Exception as e:
            print(
                f"[WARN] Detection failed on {video_name}, frame {frame_idx}: {e}")
            frame_idx += 1
            continue

        img_h, img_w = frame.shape[:2]

        candidate_boxes = []
        candidate_scores = []
        candidate_labels = []

        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            box = [float(v) for v in box.tolist()]
            ok, reason = valid_box(box, img_w, img_h)

            if not ok:
                stats["boxes_filtered_invalid"] += 1
                stats["reject_reasons"][reason] = stats["reject_reasons"].get(
                    reason, 0) + 1
                continue

            candidate_boxes.append(clip_box(box, img_w, img_h))
            candidate_scores.append(float(score))
            candidate_labels.append(str(label))

        if len(candidate_boxes) == 0:
            stats["frames_skipped_no_boxes"] += 1
            skipped_no_boxes += 1
            frame_idx += 1
            continue

        # Шаг 1: NMS
        keep_idx = nms_boxes(
            candidate_boxes, candidate_scores, NMS_IOU_THRESHOLD)
        stats["boxes_filtered_nms"] += len(candidate_boxes) - len(keep_idx)

        nms_boxes_kept = [candidate_boxes[i] for i in keep_idx]
        nms_scores_kept = [candidate_scores[i] for i in keep_idx]
        nms_labels_kept = [candidate_labels[i] for i in keep_idx]

        # Шаг 2: удаление почти вложенных боксов
        final_boxes, final_scores, final_labels, removed_contained = remove_contained_boxes(
            nms_boxes_kept,
            nms_scores_kept,
            nms_labels_kept,
            containment_threshold=CONTAINMENT_THRESHOLD
        )
        stats["boxes_filtered_contained"] += removed_contained

        if len(final_boxes) == 0:
            stats["frames_skipped_no_boxes"] += 1
            skipped_no_boxes += 1
            frame_idx += 1
            continue

        yolo_lines = []
        for box in final_boxes:
            xc, yc, bw, bh = xyxy_to_yolo(box, img_w, img_h)
            yolo_lines.append(
                f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        out_stem = f"{video_stem}_frame_{frame_idx:06d}"
        img_out_path = IMAGES_ALL_DIR / f"{out_stem}.jpg"
        txt_out_path = LABELS_ALL_DIR / f"{out_stem}.txt"
        vis_out_path = VIS_DIR / f"{out_stem}_vis.jpg"
        meta_out_path = META_DIR / f"{out_stem}.json"

        cv2.imwrite(str(img_out_path), frame)
        txt_out_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        meta = {
            "source_video": video_name,
            "frame_idx": frame_idx,
            "brightness": brightness,
            "sharpness": sharpness,
            "prompt": TEXT_PROMPT,
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
            "raw_detections": len(results["boxes"]),
            "after_size_shape_filter": len(candidate_boxes),
            "after_nms": len(nms_boxes_kept),
            "after_containment_filter": len(final_boxes),
            "detections": [
                {
                    "label": label,
                    "score": score,
                    "bbox_xyxy": box
                }
                for box, score, label in zip(final_boxes, final_scores, final_labels)
            ]
        }
        meta_out_path.write_text(json.dumps(
            meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if SAVE_VIS:
            save_visualization(frame, final_boxes,
                               final_scores, final_labels, vis_out_path)

        for box, score in zip(final_boxes, final_scores):
            xc, yc, bw, bh = xyxy_to_yolo(box, img_w, img_h)
            stats.setdefault("box_scores", []).append(float(score))
            stats.setdefault("box_widths", []).append(float(bw))
            stats.setdefault("box_heights", []).append(float(bh))
            stats.setdefault("box_areas", []).append(float(bw * bh))
            stats.setdefault("boxes_per_frame", []).append(len(final_boxes))

        stats["frames_labeled"] += 1
        saved_from_video += 1
        frame_idx += 1

    cap.release()

    stats["video_stats"][video_name] = {
        "saved_labeled_frames": saved_from_video,
        "skipped_quality": skipped_quality,
        "skipped_no_boxes": skipped_no_boxes
    }

    print(f"Saved labeled frames from {video_name}: {saved_from_video}")


# =========================================================
# TRAIN / VAL SPLIT
# =========================================================


all_images = [
    p for p in IMAGES_ALL_DIR.glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
]


random.seed(RANDOM_SEED)
random.shuffle(all_images)


if len(all_images) == 0:
    raise RuntimeError(
        "No labeled images were produced. Check thresholds/prompt/video orientation.")


val_count = max(1, int(len(all_images) * VAL_RATIO)
                ) if len(all_images) > 1 else 0
val_set = set(all_images[:val_count])
train_set = set(all_images[val_count:])


def copy_split(image_paths, split):
    for img_path in image_paths:
        stem = img_path.stem
        label_path = LABELS_ALL_DIR / f"{stem}.txt"

        if not label_path.exists():
            continue

        if split == "train":
            img_dst = IMAGES_TRAIN_DIR / img_path.name
            lbl_dst = LABELS_TRAIN_DIR / label_path.name
        else:
            img_dst = IMAGES_VAL_DIR / img_path.name
            lbl_dst = LABELS_VAL_DIR / label_path.name

        shutil.copy2(img_path, img_dst)
        shutil.copy2(label_path, lbl_dst)


copy_split(train_set, "train")
copy_split(val_set, "val")


# =========================================================
# WRITE YOLO YAML
# =========================================================


yaml_path = YOLO_ROOT / "data.yaml"
yaml_text = f"""
path: {YOLO_ROOT.resolve()}
train: images/train
val: images/val


nc: 1
names:
  0: {CLASS_NAME}
""".strip()


yaml_path.write_text(yaml_text, encoding="utf-8")


# =========================================================
# SAVE GLOBAL STATS
# =========================================================


stats_path = YOLO_ROOT / "pipeline_stats.json"
stats_path.write_text(json.dumps(
    stats, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================================================
# FINAL REPORT
# =========================================================


print("\\n=== DONE ===")
print(json.dumps(stats, ensure_ascii=False, indent=2))
print("YOLO dataset:", str(YOLO_ROOT.resolve()))
print("data.yaml:", str(yaml_path.resolve()))
print("train images:", len(list(IMAGES_TRAIN_DIR.glob("*"))))
print("val images:", len(list(IMAGES_VAL_DIR.glob("*"))))
print("train labels:", len(list(LABELS_TRAIN_DIR.glob("*.txt"))))
print("val labels:", len(list(LABELS_VAL_DIR.glob("*.txt"))))
