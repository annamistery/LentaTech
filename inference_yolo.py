from logging_setup import setup_logging
from config import CFG
from ultralytics import YOLO
import torch
import cv2
from pathlib import Path
import time
import sys
import argparse
import os

# =========================================================
# BLOCK YOLO NETWORK CALLS
# Better set before ultralytics import
# =========================================================
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["YOLO_SYNC"] = "0"
os.environ["YOLO_CHECK_UPDATE"] = "0"
os.environ["YOLO_OFFLINE"] = "1"


logger = setup_logging("inference_yolo")


WINDOW_NAME = "Price Tag Detector — Q to quit"


def draw_box(frame, x1, y1, x2, y2, label, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), CFG.BOX_COLOR, 2)

    text = f"{label} {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    pad = 3
    bx1, by1 = x1, max(0, y1 - th - baseline - pad * 2)
    bx2, by2 = x1 + tw + pad * 2, y1
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), CFG.BG_COLOR, -1)

    cv2.putText(
        frame,
        text,
        (bx1 + pad, by2 - baseline - 1),
        font,
        scale,
        CFG.TEXT_COLOR,
        thick,
        cv2.LINE_AA,
    )


def draw_hud(frame, fps, total_det, frame_idx):
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
        cv2.putText(frame, line, (x + 1, y + 1), font,
                    scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), font, scale,
                    (220, 220, 220), thick, cv2.LINE_AA)


def resolve_video_source(video_path):
    return int(video_path) if str(video_path).isdigit() else str(video_path)


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def run_inference(
    model_path,
    video_path,
    output_dir,
    conf: float = CFG.CONF_THRESH,
    iou: float = CFG.IOU_THRESH,
    imgsz: int = CFG.IMGSZ,
    save_video: bool = CFG.SAVE_VIDEO,
    show: bool = CFG.SHOW_WINDOW,
    skip: int = CFG.SKIP_FRAMES,
):
    model_path = Path(model_path) if not str(
        model_path).isdigit() else model_path
    output_dir = Path(output_dir)

    if isinstance(model_path, Path) and not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device : %s", device)
    logger.info("Model  : %s", model_path)
    logger.info("Source : %s", video_path)

    model = YOLO(str(model_path))
    model.to(device)
    class_names = model.names

    source = resolve_video_source(video_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("Resolution : %sx%s | src FPS : %.1f | total frames : %s",
                fw, fh, src_fps, total_frames)

    writer = None
    out_path = None

    try:
        if save_video:
            ensure_output_dir(output_dir)
            stem = Path(str(video_path)).stem if not str(
                video_path).isdigit() else "webcam"
            out_path = output_dir / f"{stem}_detected.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (fw, fh))

            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {out_path}")

            logger.info("Output: %s", out_path)

        frame_idx = 0
        total_dets = 0
        fps_buffer = []

        logger.info("Running... Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if skip > 0 and (frame_idx % (skip + 1)) != 0:
                if writer:
                    writer.write(frame)
                continue

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

            inf_time = t1 - t0
            fps_buffer.append(1.0 / max(inf_time, 1e-6))
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            fps = sum(fps_buffer) / len(fps_buffer)

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

            draw_hud(frame, fps, det_count, frame_idx)

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Stopped by user.")
                    break

            if frame_idx % 100 == 0:
                pct = (frame_idx / total_frames *
                       100) if total_frames > 0 else 0
                logger.info(
                    "frame %6d / %s (%5.1f%%) FPS %5.1f dets_this_frame %d",
                    frame_idx,
                    total_frames,
                    pct,
                    fps,
                    det_count,
                )

        logger.info("=== DONE ===")
        logger.info("Frames processed : %d", frame_idx)
        logger.info("Total detections : %d", total_dets)
        if out_path:
            logger.info("Saved video      : %s", out_path)

        print("\n=== DONE ===")
        print(f"Frames processed : {frame_idx}")
        print(f"Total detections : {total_dets}")
        if out_path:
            print(f"Saved video      : {out_path}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Price Tag Inference (YOLO12)")
    p.add_argument("--model", default=str(CFG.INFERENCE_MODEL_PATH),
                   help="Path to model .pt")
    p.add_argument("--video", default=str(CFG.INFERENCE_VIDEO_PATH),
                   help="Path to video or 0 for webcam")
    p.add_argument(
        "--output", default=str(CFG.INFERENCE_OUTPUT_DIR), help="Output folder")
    p.add_argument("--conf", default=CFG.CONF_THRESH,
                   type=float, help="Confidence threshold")
    p.add_argument("--iou", default=CFG.IOU_THRESH,
                   type=float, help="NMS IoU threshold")
    p.add_argument("--imgsz", default=CFG.IMGSZ,
                   type=int, help="Inference image size")
    p.add_argument("--skip", default=CFG.SKIP_FRAMES, type=int,
                   help="Skip N frames between inferences")
    p.add_argument("--no-save", action="store_true",
                   help="Don't save output video")
    p.add_argument("--no-show", action="store_true",
                   help="Don't show window (headless)")

    args, unknown = p.parse_known_args()
    if unknown:
        logger.info("Ignored unknown args: %s", unknown)
        print(f"[INFO] Ignored unknown args: {unknown}", file=sys.stderr)

    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        run_inference(
            model_path=args.model,
            video_path=args.video,
            output_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save_video=not args.no_save,
            show=not args.no_show,
            skip=args.skip,
        )
    except Exception:
        logger.exception("Fatal error in inference_yolo.py")
        raise
