import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from config import CFG
from logging_setup import setup_logging


logger = setup_logging("dino")


def ensure_dirs() -> None:
    for p in [
        CFG.WORK_DIR,
        CFG.YOLO_ROOT,
        CFG.IMAGES_ALL_DIR,
        CFG.LABELS_ALL_DIR,
        CFG.IMAGES_TRAIN_DIR,
        CFG.IMAGES_VAL_DIR,
        CFG.LABELS_TRAIN_DIR,
        CFG.LABELS_VAL_DIR,
        CFG.VIS_DIR,
        CFG.META_DIR,
        CFG.LOGS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def load_local_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Loading local model from: %s", CFG.GROUNDING_DINO_DIR)

    try:
        processor = AutoProcessor.from_pretrained(
            str(CFG.GROUNDING_DINO_DIR),
            local_files_only=True
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(CFG.GROUNDING_DINO_DIR),
            local_files_only=True
        ).to(device)
        model.eval()
    except Exception:
        logger.exception("Failed to load local Grounding DINO model")
        raise

    logger.info("Local Grounding DINO loaded successfully")
    return processor, model, device


def is_good_frame(
    frame_bgr: np.ndarray,
    min_brightness: float,
    min_laplacian_var: float
) -> Tuple[bool, float, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_good = brightness >= min_brightness and sharpness >= min_laplacian_var
    return is_good, brightness, sharpness


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if CFG.ROTATE_180_BEFORE_FLIP:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if CFG.HORIZONTAL_FLIP:
        frame_bgr = cv2.flip(frame_bgr, 1)
    return frame_bgr


def detect_price_tags(
    pil_image: Image.Image,
    processor,
    model,
    device: str
) -> Dict[str, Any]:
    inputs = processor(
        images=pil_image,
        text=CFG.TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=CFG.BOX_THRESHOLD,
        text_threshold=CFG.TEXT_THRESHOLD,
        target_sizes=[(pil_image.height, pil_image.width)]
    )[0]

    return results


def xyxy_to_yolo(box: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return xc, yc, bw, bh


def clip_box(box: List[float], img_w: int, img_h: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), img_w - 1))
    y1 = max(0.0, min(float(y1), img_h - 1))
    x2 = max(0.0, min(float(x2), img_w - 1))
    y2 = max(0.0, min(float(y2), img_h - 1))
    return [x1, y1, x2, y2]


def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def containment_ratio(inner: List[float], outer: List[float]) -> float:
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_inner = box_area(inner)
    return inter / area_inner if area_inner > 0 else 0.0


def valid_box(box: List[float], img_w: int, img_h: int) -> Tuple[bool, str]:
    x1, y1, x2, y2 = clip_box(box, img_w, img_h)
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False, "non_positive"

    if w < CFG.MIN_BOX_WIDTH_PX or h < CFG.MIN_BOX_HEIGHT_PX:
        return False, "too_small_px"

    wn = w / img_w
    hn = h / img_h
    area_n = (w * h) / (img_w * img_h)
    aspect = w / h

    if wn < CFG.MIN_BOX_WIDTH_NORM or hn < CFG.MIN_BOX_HEIGHT_NORM:
        return False, "too_small_norm"

    if wn > CFG.MAX_BOX_WIDTH_NORM or hn > CFG.MAX_BOX_HEIGHT_NORM:
        return False, "too_large_norm"

    if area_n < CFG.MIN_BOX_AREA_NORM:
        return False, "too_small_area"

    if area_n > CFG.MAX_BOX_AREA_NORM:
        return False, "too_large_area"

    if aspect < CFG.MIN_ASPECT_RATIO or aspect > CFG.MAX_ASPECT_RATIO:
        return False, "bad_aspect"

    if max(wn, hn) > CFG.MAX_EDGE_SHARE:
        return False, "too_large_edge"

    return True, "ok"


def nms_boxes(boxes: List[List[float]], scores: List[float], iou_threshold: float) -> List[int]:
    if not boxes:
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


def remove_contained_boxes(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[str],
    containment_threshold: float
) -> Tuple[List[List[float]], List[float], List[str], int]:
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

            if ci >= containment_threshold:
                if scores[i] <= scores[j]:
                    keep[i] = False
                    removed += 1
                    break

            if cj >= containment_threshold and scores[j] < scores[i]:
                keep[j] = False
                removed += 1

    new_boxes = [b for b, k in zip(boxes, keep) if k]
    new_scores = [s for s, k in zip(scores, keep) if k]
    new_labels = [l for l, k in zip(labels, keep) if k]
    return new_boxes, new_scores, new_labels, removed


def save_visualization(
    frame_bgr: np.ndarray,
    boxes: List[List[float]],
    scores: List[float],
    labels: List[str],
    out_path: Path
) -> None:
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

    ok = cv2.imwrite(str(out_path), vis)
    if not ok:
        raise IOError(f"Failed to save visualization: {out_path}")


def find_videos() -> List[Path]:
    if not CFG.VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video directory does not exist: {CFG.VIDEO_DIR}")

    video_files = [
        p for p in sorted(CFG.VIDEO_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in CFG.VIDEO_EXTS
    ]

    logger.info("Found videos: %d", len(video_files))
    for vf in video_files:
        logger.info(" - %s", vf.name)

    if not video_files:
        raise FileNotFoundError(f"No video files found in: {CFG.VIDEO_DIR}")

    return video_files


def init_stats(videos_total: int) -> Dict[str, Any]:
    return {
        "videos_total": videos_total,
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
        "video_stats": {},
        "box_scores": [],
        "box_widths": [],
        "box_heights": [],
        "box_areas": [],
        "boxes_per_frame": [],
    }


def process_video(video_path: Path, processor, model, device: str, stats: Dict[str, Any]) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path.name)
        return

    frame_idx = 0
    saved_from_video = 0
    skipped_quality = 0
    skipped_no_boxes = 0
    video_stem = video_path.stem

    logger.info("Processing: %s", video_path.name)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            stats["frames_seen"] += 1

            if frame_idx % CFG.FRAME_STEP != 0:
                frame_idx += 1
                continue

            stats["frames_sampled"] += 1
            frame = preprocess_frame(frame)

            good, brightness, sharpness = is_good_frame(
                frame,
                min_brightness=CFG.MIN_BRIGHTNESS,
                min_laplacian_var=CFG.MIN_LAPLACIAN_VAR
            )

            if not good:
                stats["frames_skipped_quality"] += 1
                skipped_quality += 1
                frame_idx += 1
                continue

            stats["frames_good"] += 1

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                results = detect_price_tags(
                    pil_image, processor, model, device)
            except Exception:
                logger.exception(
                    "Detection failed on %s, frame %d", video_path.name, frame_idx)
                frame_idx += 1
                continue

            img_h, img_w = frame.shape[:2]

            candidate_boxes: List[List[float]] = []
            candidate_scores: List[float] = []
            candidate_labels: List[str] = []

            try:
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
            except Exception:
                logger.exception(
                    "Postprocess failed on %s, frame %d", video_path.name, frame_idx)
                frame_idx += 1
                continue

            if len(candidate_boxes) == 0:
                stats["frames_skipped_no_boxes"] += 1
                skipped_no_boxes += 1
                frame_idx += 1
                continue

            keep_idx = nms_boxes(
                candidate_boxes, candidate_scores, CFG.NMS_IOU_THRESHOLD)
            stats["boxes_filtered_nms"] += len(candidate_boxes) - len(keep_idx)

            nms_boxes_kept = [candidate_boxes[i] for i in keep_idx]
            nms_scores_kept = [candidate_scores[i] for i in keep_idx]
            nms_labels_kept = [candidate_labels[i] for i in keep_idx]

            final_boxes, final_scores, final_labels, removed_contained = remove_contained_boxes(
                nms_boxes_kept,
                nms_scores_kept,
                nms_labels_kept,
                containment_threshold=CFG.CONTAINMENT_THRESHOLD
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
                    f"{CFG.CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            out_stem = f"{video_stem}_frame_{frame_idx:06d}"
            img_out_path = CFG.IMAGES_ALL_DIR / f"{out_stem}.jpg"
            txt_out_path = CFG.LABELS_ALL_DIR / f"{out_stem}.txt"
            vis_out_path = CFG.VIS_DIR / f"{out_stem}_vis.jpg"
            meta_out_path = CFG.META_DIR / f"{out_stem}.json"

            meta = {
                "source_video": video_path.name,
                "frame_idx": frame_idx,
                "brightness": brightness,
                "sharpness": sharpness,
                "prompt": CFG.TEXT_PROMPT,
                "box_threshold": CFG.BOX_THRESHOLD,
                "text_threshold": CFG.TEXT_THRESHOLD,
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

            try:
                ok = cv2.imwrite(str(img_out_path), frame)
                if not ok:
                    raise IOError(f"Failed to save image: {img_out_path}")

                txt_out_path.write_text(
                    "\n".join(yolo_lines) + "\n", encoding="utf-8")
                meta_out_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

                if CFG.SAVE_VIS:
                    save_visualization(
                        frame, final_boxes, final_scores, final_labels, vis_out_path)
            except Exception:
                logger.exception(
                    "Failed saving outputs on %s, frame %d", video_path.name, frame_idx)
                frame_idx += 1
                continue

            for box, score in zip(final_boxes, final_scores):
                xc, yc, bw, bh = xyxy_to_yolo(box, img_w, img_h)
                stats["box_scores"].append(float(score))
                stats["box_widths"].append(float(bw))
                stats["box_heights"].append(float(bh))
                stats["box_areas"].append(float(bw * bh))
                stats["boxes_per_frame"].append(len(final_boxes))

            stats["frames_labeled"] += 1
            saved_from_video += 1
            frame_idx += 1

    finally:
        cap.release()

    stats["video_stats"][video_path.name] = {
        "saved_labeled_frames": saved_from_video,
        "skipped_quality": skipped_quality,
        "skipped_no_boxes": skipped_no_boxes
    }

    logger.info("Saved labeled frames from %s: %d",
                video_path.name, saved_from_video)


def copy_split(image_paths: List[Path], split: str) -> int:
    copied = 0

    for img_path in image_paths:
        stem = img_path.stem
        label_path = CFG.LABELS_ALL_DIR / f"{stem}.txt"

        if not label_path.exists():
            logger.warning("Missing label for image: %s", img_path.name)
            continue

        if split == "train":
            img_dst = CFG.IMAGES_TRAIN_DIR / img_path.name
            lbl_dst = CFG.LABELS_TRAIN_DIR / label_path.name
        else:
            img_dst = CFG.IMAGES_VAL_DIR / img_path.name
            lbl_dst = CFG.LABELS_VAL_DIR / label_path.name

        try:
            shutil.copy2(img_path, img_dst)
            shutil.copy2(label_path, lbl_dst)
            copied += 1
        except Exception:
            logger.exception("Failed to copy %s into %s split",
                             img_path.name, split)

    return copied


def write_yaml() -> Path:
    yaml_path = CFG.YOLO_ROOT / "data.yaml"
    yaml_text = f"""
path: {CFG.YOLO_ROOT.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: {CFG.CLASS_NAME}
""".strip()

    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def save_stats(stats: Dict[str, Any]) -> Path:
    stats_path = CFG.YOLO_ROOT / "pipeline_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return stats_path


def main() -> None:
    ensure_dirs()

    processor, model, device = load_local_model()
    video_files = find_videos()
    stats = init_stats(len(video_files))

    for video_path in video_files:
        try:
            process_video(video_path, processor, model, device, stats)
        except Exception:
            logger.exception(
                "Unhandled error while processing %s", video_path.name)

    all_images = [
        p for p in CFG.IMAGES_ALL_DIR.glob("*")
        if p.suffix.lower() in CFG.IMAGE_EXTS
    ]

    random.seed(CFG.RANDOM_SEED)
    random.shuffle(all_images)

    if len(all_images) == 0:
        raise RuntimeError(
            "No labeled images were produced. Check thresholds/prompt/video orientation."
        )

    val_count = max(1, int(len(all_images) * CFG.VAL_RATIO)
                    ) if len(all_images) > 1 else 0
    val_set = set(all_images[:val_count])
    train_set = set(all_images[val_count:])

    train_copied = copy_split(list(train_set), "train")
    val_copied = copy_split(list(val_set), "val")

    yaml_path = write_yaml()
    stats_path = save_stats(stats)

    logger.info("=== DONE ===")
    logger.info("Stats saved to: %s", stats_path.resolve())
    logger.info("YOLO dataset: %s", CFG.YOLO_ROOT.resolve())
    logger.info("data.yaml: %s", yaml_path.resolve())
    logger.info("train images copied: %d", train_copied)
    logger.info("val images copied: %d", val_copied)
    logger.info("train images total: %d", len(
        list(CFG.IMAGES_TRAIN_DIR.glob("*"))))
    logger.info("val images total: %d", len(
        list(CFG.IMAGES_VAL_DIR.glob("*"))))
    logger.info("train labels total: %d", len(
        list(CFG.LABELS_TRAIN_DIR.glob("*.txt"))))
    logger.info("val labels total: %d", len(
        list(CFG.LABELS_VAL_DIR.glob("*.txt"))))

    print("\n=== DONE ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("YOLO dataset:", str(CFG.YOLO_ROOT.resolve()))
    print("data.yaml:", str(yaml_path.resolve()))
    print("train images:", len(list(CFG.IMAGES_TRAIN_DIR.glob("*"))))
    print("val images:", len(list(CFG.IMAGES_VAL_DIR.glob("*"))))
    print("train labels:", len(list(CFG.LABELS_TRAIN_DIR.glob("*.txt"))))
    print("val labels:", len(list(CFG.LABELS_VAL_DIR.glob("*.txt"))))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in dino.py")
        raise
