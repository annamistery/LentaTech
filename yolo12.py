import json
import os
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO, settings

from config import CFG
from logging_setup import setup_logging


# =========================================================
# BLOCK ALL YOLO NETWORK CALLS
# Must be set before training logic
# =========================================================
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["YOLO_SYNC"] = "0"
os.environ["YOLO_CHECK_UPDATE"] = "0"
os.environ["YOLO_HUB_NO_UPLOAD"] = "1"
os.environ["YOLO_OFFLINE"] = "1"


logger = setup_logging("yolo12_train")


def ensure_dirs() -> None:
    CFG.YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CFG.LOGS_DIR.mkdir(parents=True, exist_ok=True)


def configure_ultralytics_settings() -> None:
    settings.update(
        {
            "datasets_dir": str(CFG.YOLO_ROOT),
            "weights_dir": str(CFG.YOLO_WEIGHTS_DIR),
            "runs_dir": str(CFG.YOLO_RUNS_DIR),
            "sync": False,
        }
    )
    logger.info("Ultralytics settings updated")
    logger.info("datasets_dir: %s", CFG.YOLO_ROOT)
    logger.info("weights_dir: %s", CFG.YOLO_WEIGHTS_DIR)
    logger.info("runs_dir: %s", CFG.YOLO_RUNS_DIR)


def rewrite_data_yaml() -> Path:
    yaml_text = f"""path: {CFG.YOLO_ROOT.as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: {CFG.CLASS_NAME}
"""
    CFG.DATA_YAML.write_text(yaml_text, encoding="utf-8")
    logger.info("data.yaml rewritten at: %s", CFG.DATA_YAML)
    return CFG.DATA_YAML


def validate_dataset() -> dict:
    required_paths = [
        CFG.DATA_YAML,
        CFG.IMAGES_TRAIN_DIR,
        CFG.IMAGES_VAL_DIR,
        CFG.LABELS_TRAIN_DIR,
        CFG.LABELS_VAL_DIR,
    ]

    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    if not CFG.YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local YOLO model not found: {CFG.YOLO_MODEL_PATH}\n"
            f"Put {CFG.YOLO_MODEL_FILENAME} into: {CFG.YOLO_WEIGHTS_DIR}"
        )

    train_images = list(CFG.IMAGES_TRAIN_DIR.glob("*.*"))
    val_images = list(CFG.IMAGES_VAL_DIR.glob("*.*"))
    train_labels = list(CFG.LABELS_TRAIN_DIR.glob("*.txt"))
    val_labels = list(CFG.LABELS_VAL_DIR.glob("*.txt"))

    logger.info("Dataset check:")
    logger.info("  data.yaml     : %s", CFG.DATA_YAML)
    logger.info("  model         : %s", CFG.YOLO_MODEL_PATH)
    logger.info("  device        : %s",
                "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  workers       : %d", CFG.WORKERS)
    logger.info("  train images  : %d", len(train_images))
    logger.info("  val images    : %d", len(val_images))
    logger.info("  train labels  : %d", len(train_labels))
    logger.info("  val labels    : %d", len(val_labels))

    if len(train_images) == 0:
        raise RuntimeError("No train images found.")
    if len(val_images) == 0:
        raise RuntimeError("No val images found.")

    return {
        "train_images": train_images,
        "val_images": val_images,
        "train_labels": train_labels,
        "val_labels": val_labels,
    }


def resolve_save_dir(fallback_name: str) -> Optional[Path]:
    fallback = CFG.YOLO_RUNS_DIR / fallback_name
    if fallback.exists():
        logger.info("Using fallback save_dir: %s", fallback.resolve())
        return fallback.resolve()

    subdirs = [p for p in CFG.YOLO_RUNS_DIR.glob("*") if p.is_dir()]
    if subdirs:
        latest = sorted(subdirs, key=lambda x: x.stat().st_mtime,
                        reverse=True)[0].resolve()
        logger.info("Auto-detected save_dir: %s", latest)
        return latest

    return None


def train_model(device: str) -> Optional[Path]:
    logger.info("Loading local YOLO model: %s", CFG.YOLO_MODEL_PATH)
    model = YOLO(str(CFG.YOLO_MODEL_PATH))

    save_dir = None
    train_results = None

    try:
        train_results = model.train(
            data=str(CFG.DATA_YAML),
            epochs=CFG.EPOCHS,
            imgsz=CFG.IMGSZ,
            batch=CFG.BATCH,
            device=device,
            workers=CFG.WORKERS,
            project=str(CFG.YOLO_RUNS_DIR),
            name=CFG.YOLO_EXPERIMENT_NAME,
            patience=CFG.PATIENCE,
            save=True,
            save_period=CFG.SAVE_PERIOD,
            verbose=True,
            val=True,
            pretrained=False,
            degrees=3.0,
            translate=0.08,
            scale=0.20,
            shear=1.0,
            perspective=0.0005,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.010,
            hsv_s=0.35,
            hsv_v=0.25,
            mosaic=0.15,
            mixup=0.0,
            copy_paste=0.0,
        )

        if train_results is not None and hasattr(train_results, "save_dir"):
            sd = train_results.save_dir
            save_dir = Path(sd).resolve() if isinstance(
                sd, (str, Path)) else None

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
    except Exception:
        logger.exception("Training failed")

    if save_dir is None:
        save_dir = resolve_save_dir(CFG.YOLO_EXPERIMENT_NAME)

    return save_dir


def select_validation_model(save_dir: Path) -> Path:
    best_model_path = save_dir / "weights" / "best.pt"
    last_model_path = save_dir / "weights" / "last.pt"

    if best_model_path.exists():
        logger.info("Using best.pt for validation: %s", best_model_path)
        return best_model_path

    if last_model_path.exists():
        logger.warning("best.pt not found. Using last.pt: %s", last_model_path)
        return last_model_path

    logger.warning(
        "No trained weights found. Using initial model: %s", CFG.YOLO_MODEL_PATH)
    return CFG.YOLO_MODEL_PATH


def validate_model(val_model_path: Path, device: str):
    val_model = YOLO(str(val_model_path))
    return val_model.val(
        data=str(CFG.DATA_YAML),
        split="val",
        imgsz=CFG.IMGSZ,
        batch=CFG.BATCH,
        device=device,
        workers=CFG.WORKERS,
    )


def save_summary(
    dataset_info: dict,
    save_dir: Path,
    val_model_path: Path,
    val_metrics,
    device: str,
) -> Path:
    best_model_path = save_dir / "weights" / "best.pt"
    last_model_path = save_dir / "weights" / "last.pt"

    map50 = float(getattr(val_metrics.box, "map50", 0.0))
    map50_95 = float(getattr(val_metrics.box, "map", 0.0))
    map75 = float(getattr(val_metrics.box, "map75", 0.0))

    summary = {
        "project_root": str(CFG.BASE_DIR),
        "dataset_root": str(CFG.YOLO_ROOT),
        "data_yaml": str(CFG.DATA_YAML),
        "initial_model_path": str(CFG.YOLO_MODEL_PATH),
        "save_dir": str(save_dir),
        "val_model_used": str(val_model_path),
        "best_model_path": str(best_model_path) if best_model_path.exists() else None,
        "last_model_path": str(last_model_path) if last_model_path.exists() else None,
        "device": device,
        "workers": CFG.WORKERS,
        "epochs_configured": CFG.EPOCHS,
        "train_images": len(dataset_info["train_images"]),
        "val_images": len(dataset_info["val_images"]),
        "train_labels": len(dataset_info["train_labels"]),
        "val_labels": len(dataset_info["val_labels"]),
        "map50": map50,
        "map50_95": map50_95,
        "map75": map75,
    }

    summary_path = save_dir / "training_summary.json"
    summary_path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=== TRAINING COMPLETE ===")
    logger.info("Summary saved: %s", summary_path)
    logger.info("Best weights : %s",
                best_model_path if best_model_path.exists() else "— not found")
    logger.info("Last weights : %s",
                last_model_path if last_model_path.exists() else "— not found")

    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nBest weights :",
          best_model_path if best_model_path.exists() else "— not found")
    print("Last weights  :", last_model_path if last_model_path.exists()
          else "— not found")
    print("Summary saved :", summary_path)

    return summary_path


def main() -> None:
    ensure_dirs()
    configure_ultralytics_settings()
    rewrite_data_yaml()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_info = validate_dataset()

    save_dir = train_model(device=device)
    if save_dir is None:
        logger.warning(
            "Could not determine save_dir. Check runs directory manually.")
        return

    val_model_path = select_validation_model(save_dir)
    val_metrics = validate_model(val_model_path=val_model_path, device=device)

    save_summary(
        dataset_info=dataset_info,
        save_dir=save_dir,
        val_model_path=val_model_path,
        val_metrics=val_metrics,
        device=device,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in yolo12.py")
        raise
