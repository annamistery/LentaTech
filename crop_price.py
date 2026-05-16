import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# =========================================================
# БЛОКИРОВКА СЕТЕВЫХ ЗАПРОСОВ
# =========================================================
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["YOLO_SYNC"] = "0"
os.environ["YOLO_CHECK_UPDATE"] = "0"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_VERBOSE"] = "0"

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = r"E:\Data Science\Хакатон_ценники\runs_yolo12_640\price_tags_yolo12n_img640_cleanv1\weights\best.pt"
# папка с видео или изображениями
SOURCE_DIR = r"E:\Data Science\Хакатон_ценники\video_price"
# куда сохранять вырезанные ценники
OUTPUT_DIR = r"E:\Data Science\Хакатон_ценники\crops"

CONF_THRESH = 0.40
IOU_THRESH = 0.45
IMGSZ = 640

# Отступ вокруг бокса в пикселях (чтобы не обрезать край ценника)
PADDING = 4

# Минимальный размер вырезанного ценника в пикселях
MIN_CROP_W = 20
MIN_CROP_H = 12

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# Если обрабатываем видео — брать каждый N-й кадр
FRAME_STEP = 10

# =========================================================
# INIT
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

model = YOLO(str(MODEL_PATH))
model.to(device)

out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

total_crops = 0
total_frames = 0

# =========================================================
# HELPERS
# =========================================================


def crop_and_save(frame_bgr, boxes, source_stem, frame_idx, crop_counter):
    """Вырезает боксы из кадра и сохраняет как отдельные файлы."""
    h, w = frame_bgr.shape[:2]
    saved = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu())

        # Добавляем отступ и клипируем по границам кадра
        x1 = max(0, int(x1) - PADDING)
        y1 = max(0, int(y1) - PADDING)
        x2 = min(w, int(x2) + PADDING)
        y2 = min(h, int(y2) + PADDING)

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Пропускаем слишком маленькие кропы
        if crop_w < MIN_CROP_W or crop_h < MIN_CROP_H:
            continue

        crop = frame_bgr[y1:y2, x1:x2]

        # Имя файла: источник_кадр_номер_conf.jpg
        filename = f"{source_stem}_frame{frame_idx:06d}_box{i:02d}_conf{conf:.2f}.jpg"
        out_path = out_dir / filename

        cv2.imwrite(str(out_path), crop)
        crop_counter += 1
        saved += 1

    return crop_counter, saved


def process_image(img_path: Path):
    """Обработка одного изображения."""
    global total_crops, total_frames

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[WARN] Cannot read: {img_path.name}")
        return

    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMGSZ,
        device=device,
        verbose=False,
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return

    total_crops, saved = crop_and_save(
        frame, boxes, img_path.stem, 0, total_crops)
    total_frames += 1
    print(f"  {img_path.name}: {saved} crops")


def process_video(video_path: Path):
    """Обработка видеофайла — кадр за кадром."""
    global total_crops, total_frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path.name}")
        return

    frame_idx = 0
    video_crops = 0

    print(f"\nProcessing video: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        results = model.predict(
            source=frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMGSZ,
            device=device,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            total_crops, saved = crop_and_save(
                frame, boxes, video_path.stem, frame_idx, total_crops
            )
            video_crops += saved
            total_frames += 1

        frame_idx += 1

    cap.release()
    print(f"  Saved {video_crops} crops from {video_path.name}")


# =========================================================
# MAIN
# =========================================================
source_path = Path(SOURCE_DIR)

# Собираем все файлы
video_files = sorted([f for f in source_path.iterdir()
                     if f.suffix.lower() in VIDEO_EXTS])
image_files = sorted([f for f in source_path.iterdir()
                     if f.suffix.lower() in IMAGE_EXTS])

print(f"Found: {len(video_files)} videos, {len(image_files)} images")

# Обрабатываем изображения
for img_path in image_files:
    process_image(img_path)

# Обрабатываем видео
for video_path in video_files:
    process_video(video_path)

# =========================================================
# ИТОГ
# =========================================================
print(f"\n=== DONE ===")
print(f"Frames processed : {total_frames}")
print(f"Total crops saved: {total_crops}")
print(f"Output folder    : {out_dir.resolve()}")
print(f"\nStructure of saved files:")
print(f"  {out_dir}/")
print(f"  └── <source>_frame<N>_box<N>_conf<0.XX>.jpg")
