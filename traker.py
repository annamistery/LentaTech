import os
import cv2
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv


# =========================================================
# OFFLINE / NO-NETWORK
# =========================================================
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["YOLO_SYNC"] = "0"
os.environ["YOLO_CHECK_UPDATE"] = "0"
os.environ["YOLO_OFFLINE"] = "1"


# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = r"E:\Data Science\Хакатон_ценники\runs_yolo12_640\price_tags_yolo12s_img640_cleanv1-2\weights\best.pt"
VIDEO_PATH = r"E:\Data Science\Хакатон_ценники\26_2-10.mp4"
OUTPUT_DIR = r"E:\Data Science\Хакатон_ценники\inference_output_treker"

CONF_THRESH = 0.40
IOU_THRESH = 0.45
IMGSZ = 640
SAVE_VIDEO = True
SHOW_WINDOW = True
SKIP_FRAMES = 0

BOX_COLOR = (0, 60, 255)      # BGR
TEXT_COLOR = (255, 255, 255)  # BGR
BG_COLOR = (0, 60, 255)       # BGR

WINDOW_NAME = "YOLO + ByteTrack"


# =========================================================
# HELPERS
# =========================================================
def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def sv_color_from_bgr(bgr: tuple) -> sv.Color:
    b, g, r = bgr
    return sv.Color(r=r, g=g, b=b)


def get_tracker_ids(detections: sv.Detections) -> list:
    tracker_id = getattr(detections, "tracker_id", None)
    if tracker_id is None:
        return [-1] * len(detections)
    if isinstance(tracker_id, np.ndarray):
        return tracker_id.tolist()
    return list(tracker_id)


def get_class_names(detections: sv.Detections) -> list:
    data = getattr(detections, "data", {}) or {}
    class_names = data.get("class_name", None)

    if class_names is not None:
        if isinstance(class_names, np.ndarray):
            return class_names.tolist()
        return list(class_names)

    class_ids = getattr(detections, "class_id", None)
    if class_ids is None:
        return ["object"] * len(detections)

    if isinstance(class_ids, np.ndarray):
        class_ids = class_ids.tolist()

    return [str(c) for c in class_ids]


def get_confidences(detections: sv.Detections) -> list:
    confidence = getattr(detections, "confidence", None)
    if confidence is None:
        return [0.0] * len(detections)
    if isinstance(confidence, np.ndarray):
        return confidence.tolist()
    return list(confidence)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    current_det: int,
    frame_idx: int,
    total_frames: int,
    unique_tracks: int,
):
    elapsed_sec = frame_idx / max(fps, 1e-6)
    remaining_sec = (
        max(total_frames - frame_idx, 0) / max(fps, 1e-6)
        if total_frames > 0 else 0
    )

    lines = [
        f"FPS:        {fps:5.1f}",
        f"Frame:      {frame_idx}/{total_frames if total_frames > 0 else '?'}",
        f"Detected:   {current_det}",
        f"Tracks:     {unique_tracks}",
        f"Elapsed:    {format_time(elapsed_sec)}",
        f"Remaining:  {format_time(remaining_sec)}",
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


# =========================================================
# TRACKER ITERATOR CLASS
# =========================================================
class YOLOTracker:
    def __init__(
        self,
        model_path: str,
        video_path,
        conf: float = CONF_THRESH,
        iou: float = IOU_THRESH,
        imgsz: int = IMGSZ,
        skip: int = SKIP_FRAMES,
        device: Optional[str] = None,
    ):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.skip = max(0, int(skip))

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[YOLOTracker] Device : {self.device}")

        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        print(f"[YOLOTracker] Model  : {model_path}")

        self.tracker = sv.ByteTrack()

        source = int(video_path) if str(
            video_path).isdigit() else str(video_path)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[YOLOTracker] Cannot open video source: {video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0

        print(
            f"[YOLOTracker] {self.frame_width}x{self.frame_height} | "
            f"{self.src_fps:.1f} FPS | {self.total_frames} frames total"
        )

    def get_next_frame(self) -> Optional[Tuple[np.ndarray, sv.Detections]]:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return None

            self.frame_idx += 1

            if self.skip > 0 and (self.frame_idx % (self.skip + 1)) != 0:
                continue

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                stream=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            return frame, detections

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        self.tracker = sv.ByteTrack()
        print("[YOLOTracker] Reset to frame 0.")

    def release(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        print("[YOLOTracker] Released.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, sv.Detections]:
        result = self.get_next_frame()
        if result is None:
            raise StopIteration
        return result


# =========================================================
# MAIN
# =========================================================
def run_inference(
    model_path: str = MODEL_PATH,
    video_path=VIDEO_PATH,
    output_dir: str = OUTPUT_DIR,
    conf: float = CONF_THRESH,
    iou: float = IOU_THRESH,
    imgsz: int = IMGSZ,
    save_video: bool = SAVE_VIDEO,
    show: bool = SHOW_WINDOW,
    skip: int = SKIP_FRAMES,
):
    box_annotator = sv.BoxAnnotator(
        color=sv_color_from_bgr(BOX_COLOR)
    )
    label_annotator = sv.LabelAnnotator(
        color=sv_color_from_bgr(BG_COLOR),
        text_color=sv_color_from_bgr(TEXT_COLOR),
    )

    with YOLOTracker(
        model_path=model_path,
        video_path=video_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        skip=skip,
    ) as tracker:

        writer = None
        out_path = None

        if save_video:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(str(video_path)).stem if not str(
                video_path).isdigit() else "webcam"
            out_path = out_dir / f"{stem}_tracked.mp4"

            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                tracker.src_fps,
                (tracker.frame_width, tracker.frame_height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"[run_inference] Cannot open VideoWriter: {out_path}")

            print(f"[run_inference] Output : {out_path}")

        fps_buffer = []
        t_prev = time.perf_counter()

        total_dets_all = 0
        unique_ids = set()

        print("\nRunning... Press Q to quit.\n")

        pbar_total = tracker.total_frames if tracker.total_frames > 0 else None
        pbar = tqdm(
            total=pbar_total,
            desc="🎬 Inference",
            unit="fr",
            dynamic_ncols=True,
            colour="green",
        )

        try:
            if show:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            while True:
                result = tracker.get_next_frame()
                if result is None:
                    break

                frame, detections = result

                t_now = time.perf_counter()
                fps_buffer.append(1.0 / max(t_now - t_prev, 1e-6))
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                fps = sum(fps_buffer) / len(fps_buffer)
                t_prev = t_now

                n_det = len(detections)
                total_dets_all += n_det

                current_ids = get_tracker_ids(detections)
                unique_ids.update([tid for tid in current_ids if tid != -1])

                pbar_step = 1 if skip == 0 else (skip + 1)
                pbar.update(pbar_step)
                pbar.set_postfix(
                    {
                        "FPS": f"{fps:.1f}",
                        "dets": n_det,
                        "total_det": total_dets_all,
                        "tracks": len(unique_ids),
                    }
                )

                class_names = get_class_names(detections)
                confidences = get_confidences(detections)

                labels = [
                    f"#{tid} {cls} {conf_score:.0%}" if tid != -
                    1 else f"{cls} {conf_score:.0%}"
                    for tid, cls, conf_score in zip(current_ids, class_names, confidences)
                ]

                annotated = frame.copy()
                annotated = box_annotator.annotate(
                    scene=annotated, detections=detections)
                annotated = label_annotator.annotate(
                    scene=annotated,
                    detections=detections,
                    labels=labels,
                )

                draw_hud(
                    frame=annotated,
                    fps=fps,
                    current_det=n_det,
                    frame_idx=tracker.frame_idx,
                    total_frames=tracker.total_frames,
                    unique_tracks=len(unique_ids),
                )

                if writer is not None:
                    writer.write(annotated)

                if show:
                    cv2.imshow(WINDOW_NAME, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[run_inference] Interrupted by user.")
                        break

        finally:
            pbar.close()

            if writer is not None:
                writer.release()

            if show:
                cv2.destroyAllWindows()

        duration_sec = tracker.frame_idx / max(tracker.src_fps, 1e-6)
        avg_fps = sum(fps_buffer) / max(len(fps_buffer), 1)

        print("\n" + "=" * 50)
        print("  ✅ Готово!")
        print(
            f"  Обработано кадров : {tracker.frame_idx} / {tracker.total_frames}")
        print(f"  Длительность      : {format_time(duration_sec)}")
        print(f"  Всего детекций    : {total_dets_all}")
        print(f"  Уникальных треков : {len(unique_ids)}")
        print(f"  Средний FPS       : {avg_fps:.1f}")
        if out_path is not None:
            print(f"  Сохранено в       : {out_path}")
        print("=" * 50)

        return {
            "frames_processed": tracker.frame_idx,
            "total_frames": tracker.total_frames,
            "duration_sec": duration_sec,
            "total_detections": total_dets_all,
            "unique_tracks": len(unique_ids),
            "avg_fps": avg_fps,
            "output_path": str(out_path) if out_path is not None else None,
        }


if __name__ == "__main__":
    run_inference()
