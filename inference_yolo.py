import os
import cv2
import torch
import argparse
import time
from pathlib import Path
from ultralytics import YOLO
import sys
# =========================================================
# БЛОКИРОВКА СЕТЕВЫХ ЗАПРОСОВ
# =========================================================
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["YOLO_SYNC"] = "0"
os.environ["YOLO_CHECK_UPDATE"] = "0"
os.environ["YOLO_OFFLINE"] = "1"


# =========================================================
# CONFIG — меняй только здесь
# =========================================================
MODEL_PATH = r"E:\Data Science\Хакатон_ценники\runs_yolo12_640\price_tags_yolo12s_img640_cleanv1-2\weights\yolo12s.pt"
VIDEO_PATH = r"E:\Data Science\Хакатон_ценники\26_2-10.mp4"  # или 0 для вебкамеры
OUTPUT_DIR = r"E:\Data Science\Хакатон_ценники\inference_output"

CONF_THRESH = 0.40    # уверенность детекции
IOU_THRESH = 0.45    # NMS порог
IMGSZ = 640
SAVE_VIDEO = True    # сохранить результат в файл
SHOW_WINDOW = True    # показывать окно в реальном времени
SKIP_FRAMES = 0       # 0 = каждый кадр, 1 = каждый второй, и т.д.

# Цвет рамки и текста (BGR)
BOX_COLOR = (0, 60, 255)
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 60, 255)


# =========================================================
# OVERLAY HELPERS
# =========================================================

def draw_box(frame, x1, y1, x2, y2, label, conf):
    """Рисует аккуратную рамку с подписью и уверенностью."""
    # Рамка
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    # Подпись
    text = f"{label}  {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    # Фон подписи
    pad = 3
    bx1, by1 = x1, max(0, y1 - th - baseline - pad * 2)
    bx2, by2 = x1 + tw + pad * 2, y1
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), BG_COLOR, -1)

    # Текст
    cv2.putText(frame, text, (bx1 + pad, by2 - baseline - 1),
                font, scale, TEXT_COLOR, thick, cv2.LINE_AA)


def draw_hud(frame, fps, total_det, frame_idx):
    """HUD в левом верхнем углу — FPS, кадр, кол-во ценников."""
    lines = [
        f"FPS:       {fps:5.1f}",
        f"Frame:     {frame_idx}",
        f"Detected:  {total_det}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thick = 1
    x, y_start, dy = 10, 24, 20

    for i, line in enumerate(lines):
        y = y_start + i * dy
        cv2.putText(frame, line, (x + 1, y + 1), font, scale,
                    (0, 0, 0),   thick + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (x,     y),     font, scale,
                    (220, 220, 220), thick,  cv2.LINE_AA)


# =========================================================
# MAIN INFERENCE
# =========================================================

def run_inference(
    model_path: str,
    video_path,           # str | int (0 = webcam)
    output_dir: str,
    conf: float = CONF_THRESH,
    iou: float = IOU_THRESH,
    imgsz: int = IMGSZ,
    save_video: bool = SAVE_VIDEO,
    show: bool = SHOW_WINDOW,
    skip: int = SKIP_FRAMES,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : {model_path}")
    print(f"Source : {video_path}")

    # --- Загрузка модели ---
    model = YOLO(str(model_path))
    model.to(device)
    class_names = model.names   # {0: 'price_tag', ...}

    # --- Открытие видео ---
    source = int(video_path) if str(video_path).isdigit() else str(video_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Resolution : {fw}x{fh}  |  src FPS : {src_fps:.1f}  |  total frames : {total_frames}")

    # --- Выходной видеофайл ---
    writer = None
    out_path = None
    if save_video:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(str(video_path)).stem if not str(
            video_path).isdigit() else "webcam"
        out_path = out_dir / f"{stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (fw, fh))
        print(f"Output     : {out_path}")

    # --- Прогон ---
    frame_idx = 0
    total_dets = 0
    fps_buffer = []
    t_prev = time.perf_counter()

    print("\nRunning... Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Пропуск кадров для ускорения
        if skip > 0 and (frame_idx % (skip + 1)) != 0:
            if writer:
                writer.write(frame)
            continue

        # --- Инференс ---
        t0 = time.perf_counter()

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
            stream=False,
        )

        t1 = time.perf_counter()

        # --- FPS ---
        inf_time = t1 - t0
        fps_buffer.append(1.0 / max(inf_time, 1e-6))
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)
        fps = sum(fps_buffer) / len(fps_buffer)

        # --- Отрисовка боксов ---
        det_count = 0
        result = results[0]

        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                c_id = int(box.cls[0].cpu())
                c_val = float(box.conf[0].cpu())
                name = class_names.get(c_id, str(c_id))

                x1, y1, x2, y2 = map(int, xyxy)
                draw_box(frame, x1, y1, x2, y2, name, c_val)
                det_count += 1
                total_dets += 1

        # HUD
        draw_hud(frame, fps, det_count, frame_idx)

        # --- Запись / показ ---
        if writer:
            writer.write(frame)

        if show:
            cv2.imshow("Price Tag Detector — Q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Stopped by user.")
                break

        # Прогресс в консоль каждые 100 кадров
        if frame_idx % 100 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"  frame {frame_idx:6d} / {total_frames}  ({pct:5.1f}%)  "
                  f"FPS {fps:5.1f}  dets_this_frame {det_count}")

    # --- Финал ---
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n=== DONE ===")
    print(f"Frames processed : {frame_idx}")
    print(f"Total detections : {total_dets}")
    if out_path:
        print(f"Saved video      : {out_path}")


# =========================================================
# CLI + WINDOWS SAFE ENTRYPOINT
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Price Tag Inference (YOLO12)")
    p.add_argument("--model",    default=MODEL_PATH,   help="Path to best.pt")
    p.add_argument("--video",    default=VIDEO_PATH,
                   help="Path to video or 0 for webcam")
    p.add_argument("--output",   default=OUTPUT_DIR,   help="Output folder")
    p.add_argument("--conf",     default=CONF_THRESH,
                   type=float, help="Confidence threshold")
    p.add_argument("--iou",      default=IOU_THRESH,
                   type=float, help="NMS IoU threshold")
    p.add_argument("--imgsz",    default=IMGSZ,
                   type=int,   help="Inference image size")
    p.add_argument("--skip",     default=SKIP_FRAMES,  type=int,
                   help="Skip N frames between inferences")
    p.add_argument("--no-save",  action="store_true",
                   help="Don't save output video")
    p.add_argument("--no-show",  action="store_true",
                   help="Don't show window (headless)")
    args, unknown = p.parse_known_args()   # ← было p.parse_args()
    if unknown:
        print(f"[INFO] Ignored unknown args: {unknown}", file=sys.stderr)
    return args                            # ← возвращаем args, не кортеж


if __name__ == "__main__":
    args = parse_args()                    # ← было: args, unknown = p.parse_known_args()
    run_inference(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save_video=not args.no_save,
        show=not args.no_show,
        skip=args.skip,)
