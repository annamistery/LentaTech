from logging_setup import setup_logging
from config import CFG
from ultralytics import YOLO
import torch
import cv2
from pathlib import Path
import os

# =========================================================
# BLOCK YOLO NETWORK CALLS
# =========================================================
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["YOLO_SYNC"] = "0"
os.environ["YOLO_CHECK_UPDATE"] = "0"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_VERBOSE"] = "0"


logger = setup_logging("crop_price")


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def ensure_dirs() -> None:
    CFG.CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CFG.LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    if not CFG.CROP_MODEL_PATH.exists():
        raise FileNotFoundError(f"Crop model not found: {CFG.CROP_MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Loading model: %s", CFG.CROP_MODEL_PATH)

    model = YOLO(str(CFG.CROP_MODEL_PATH))
    model.to(device)

    return model, device


def crop_and_save(frame_bgr, boxes, source_stem: str, frame_idx: int, crop_counter: int):
    h, w = frame_bgr.shape[:2]
    saved = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu())

        x1 = max(0, int(x1) - CFG.PADDING)
        y1 = max(0, int(y1) - CFG.PADDING)
        x2 = min(w, int(x2) + CFG.PADDING)
        y2 = min(h, int(y2) + CFG.PADDING)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < CFG.MIN_CROP_W or crop_h < CFG.MIN_CROP_H:
            continue

        crop = frame_bgr[y1:y2, x1:x2]
        filename = f"{source_stem}_frame{frame_idx:06d}_box{i:02d}_conf{conf:.2f}.jpg"
        out_path = CFG.CROP_OUTPUT_DIR / filename

        ok = cv2.imwrite(str(out_path), crop)
        if not ok:
            logger.warning("Failed to save crop: %s", out_path)
            continue

        crop_counter += 1
        saved += 1

    return crop_counter, saved


def predict_frame(model, device: str, frame):
    return model.predict(
        source=frame,
        conf=CFG.CROP_CONF_THRESH,
        iou=CFG.CROP_IOU_THRESH,
        imgsz=CFG.CROP_IMGSZ,
        device=device,
        verbose=False,
    )


def process_image(img_path: Path, model, device: str, stats: dict):
    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.warning("Cannot read image: %s", img_path.name)
        return

    try:
        results = predict_frame(model, device, frame)
    except Exception:
        logger.exception("Inference failed for image: %s", img_path.name)
        return

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return

    stats["total_crops"], saved = crop_and_save(
        frame_bgr=frame,
        boxes=boxes,
        source_stem=img_path.stem,
        frame_idx=0,
        crop_counter=stats["total_crops"],
    )
    stats["total_frames"] += 1
    logger.info("%s: %d crops", img_path.name, saved)


def process_video(video_path: Path, model, device: str, stats: dict):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path.name)
        return

    frame_idx = 0
    video_crops = 0

    logger.info("Processing video: %s", video_path.name)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % CFG.FRAME_STEP != 0:
                frame_idx += 1
                continue

            try:
                results = predict_frame(model, device, frame)
            except Exception:
                logger.exception(
                    "Inference failed for %s at frame %d", video_path.name, frame_idx)
                frame_idx += 1
                continue

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                stats["total_crops"], saved = crop_and_save(
                    frame_bgr=frame,
                    boxes=boxes,
                    source_stem=video_path.stem,
                    frame_idx=frame_idx,
                    crop_counter=stats["total_crops"],
                )
                video_crops += saved
                stats["total_frames"] += 1

            frame_idx += 1

    finally:
        cap.release()

    logger.info("Saved %d crops from %s", video_crops, video_path.name)


def collect_source_files(source_path: Path):
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_path}")

    video_files = sorted([f for f in source_path.iterdir(
    ) if f.is_file() and f.suffix.lower() in VIDEO_EXTS])
    image_files = sorted([f for f in source_path.iterdir(
    ) if f.is_file() and f.suffix.lower() in IMAGE_EXTS])

    return video_files, image_files


def main():
    ensure_dirs()
    model, device = load_model()

    source_path = CFG.CROP_SOURCE_DIR
    video_files, image_files = collect_source_files(source_path)

    logger.info("Found: %d videos, %d images",
                len(video_files), len(image_files))

    stats = {
        "total_crops": 0,
        "total_frames": 0,
    }

    for img_path in image_files:
        try:
            process_image(img_path, model, device, stats)
        except Exception:
            logger.exception(
                "Unhandled error while processing image: %s", img_path.name)

    for video_path in video_files:
        try:
            process_video(video_path, model, device, stats)
        except Exception:
            logger.exception(
                "Unhandled error while processing video: %s", video_path.name)

    logger.info("=== DONE ===")
    logger.info("Frames processed : %d", stats["total_frames"])
    logger.info("Total crops saved: %d", stats["total_crops"])
    logger.info("Output folder    : %s", CFG.CROP_OUTPUT_DIR.resolve())

    print("\n=== DONE ===")
    print(f"Frames processed : {stats['total_frames']}")
    print(f"Total crops saved: {stats['total_crops']}")
    print(f"Output folder    : {CFG.CROP_OUTPUT_DIR.resolve()}")
    print("\nStructure of saved files:")
    print(f"  {CFG.CROP_OUTPUT_DIR}/")
    print("  └── <source>_frame<N>_box<N>_conf<0.XX>.jpg")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in crop_price.py")
        raise
