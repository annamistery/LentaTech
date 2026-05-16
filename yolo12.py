import os
import json
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings

# =========================================================
# БЛОКИРОВКА ВСЕХ СЕТЕВЫХ ЗАПРОСОВ YOLO
# Обязательно ДО импорта YOLO и до main()
# =========================================================
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["YOLO_SYNC"] = "0"
os.environ["YOLO_CHECK_UPDATE"] = "0"
os.environ["YOLO_HUB_NO_UPLOAD"] = "1"
os.environ["YOLO_OFFLINE"] = "1"


# =========================================================
# MAIN
# =========================================================
def main():

    # -----------------------------------------------------
    # ABSOLUTE PATHS
    # -----------------------------------------------------
    PROJECT_ROOT = Path(r"E:\Data Science\Хакатон_ценники").resolve()

    DATASET_ROOT = (PROJECT_ROOT / "work_price_tags" / "dataset").resolve()
    DATA_YAML = (DATASET_ROOT / "data.yaml").resolve()

    # yolo12s.pt, yolo12n.pt, yolo26n
    MODEL_PATH = (PROJECT_ROOT / "yolo12n.pt").resolve()
    RUNS_DIR = (PROJECT_ROOT / "runs_yolo12_640").resolve()

    # -----------------------------------------------------
    # SETTINGS — offline-режим, sync отключён
    # -----------------------------------------------------
    settings.update({
        "datasets_dir": str(DATASET_ROOT),
        "weights_dir":  str(PROJECT_ROOT),
        "runs_dir":     str(RUNS_DIR),
        "sync":         False,   # не пытаться синхронизировать с HUB
    })

    # -----------------------------------------------------
    # TRAIN PARAMS
    # -----------------------------------------------------

    MODEL_PATH = (PROJECT_ROOT / "yolo12n.pt").resolve()

    EPOCHS = 150
    IMGSZ = 640
    BATCH = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WORKERS = 0          # Windows: всегда 0, иначе нестабильно
    EXPERIMENT_NAME = "price_tags_yolo12n_img640_cleanv1"
    PATIENCE = 35
    SAVE_PERIOD = 5          # сохранять веса каждые 5 эпох для надёжности

    # -----------------------------------------------------
    # ПЕРЕЗАПИСЬ data.yaml — гарантируем правильный путь
    # (предыдущий скрипт пишет path: с resolve(), но на всякий случай
    #  перезаписываем с точными текущими путями)
    # -----------------------------------------------------
    yaml_text = f"""path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: price_tag
"""
    DATA_YAML.write_text(yaml_text, encoding="utf-8")
    print(f"data.yaml rewritten at: {DATA_YAML}")

    # -----------------------------------------------------
    # ПРОВЕРКА ДАТАСЕТА
    # -----------------------------------------------------
    required_paths = [
        DATA_YAML,
        DATASET_ROOT / "images" / "train",
        DATASET_ROOT / "images" / "val",
        DATASET_ROOT / "labels" / "train",
        DATASET_ROOT / "labels" / "val",
    ]

    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local YOLO model not found: {MODEL_PATH}\n"
            f"Put yolo12n.pt into: {PROJECT_ROOT}"
        )

    train_images = list((DATASET_ROOT / "images" / "train").glob("*.*"))
    val_images = list((DATASET_ROOT / "images" / "val").glob("*.*"))
    train_labels = list((DATASET_ROOT / "labels" / "train").glob("*.txt"))
    val_labels = list((DATASET_ROOT / "labels" / "val").glob("*.txt"))

    print("\nDataset check:")
    print("  data.yaml     :", DATA_YAML)
    print("  model         :", MODEL_PATH)
    print("  device        :", DEVICE)
    print("  workers       :", WORKERS)
    print("  train images  :", len(train_images))
    print("  val images    :", len(val_images))
    print("  train labels  :", len(train_labels))
    print("  val labels    :", len(val_labels))

    if len(train_images) == 0:
        raise RuntimeError("No train images found.")
    if len(val_images) == 0:
        raise RuntimeError("No val images found.")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # ЗАГРУЗКА ЛОКАЛЬНОЙ МОДЕЛИ
    # pretrained=False — не пытаться скачивать веса из интернета;
    # .pt файл уже содержит pretrained-веса
    # -----------------------------------------------------
    print(f"\nLoading local YOLO model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # -----------------------------------------------------
    # ОБУЧЕНИЕ С ЗАЩИТОЙ ОТ ПРЕРЫВАНИЯ
    # -----------------------------------------------------
    save_dir = None
    train_results = None

    try:
        train_results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=str(RUNS_DIR),
            name=EXPERIMENT_NAME,
            patience=PATIENCE,
            save=True,
            save_period=SAVE_PERIOD,
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

        # Безопасно достаём save_dir — в разных версиях ultralytics разный тип
        if train_results is not None and hasattr(train_results, "save_dir"):
            sd = train_results.save_dir
            save_dir = Path(sd).resolve() if isinstance(
                sd, (str, Path)) else None

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
    finally:
        # Пытаемся найти save_dir по стандартному пути YOLO даже при прерывании
        if save_dir is None:
            fallback = RUNS_DIR / EXPERIMENT_NAME
            if fallback.exists():
                save_dir = fallback.resolve()
                print(f"[INFO] Using fallback save_dir: {save_dir}")
            else:
                # Ищем последнюю созданную папку в RUNS_DIR
                subdirs = sorted(RUNS_DIR.glob(
                    "*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if subdirs:
                    save_dir = subdirs[0].resolve()
                    print(f"[INFO] Auto-detected save_dir: {save_dir}")

    if save_dir is None:
        print("[WARN] Could not determine save_dir. Check RUNS_DIR manually.")
        return

    # -----------------------------------------------------
    # ВЫБОР ЛУЧШЕЙ МОДЕЛИ ДЛЯ ВАЛИДАЦИИ
    # Приоритет: best.pt → last.pt → исходный model_path
    # -----------------------------------------------------
    best_model_path = save_dir / "weights" / "best.pt"
    last_model_path = save_dir / "weights" / "last.pt"

    if best_model_path.exists():
        val_model_path = best_model_path
        print(f"\nUsing best.pt for validation: {val_model_path}")
    elif last_model_path.exists():
        val_model_path = last_model_path
        print(f"\n[WARN] best.pt not found. Using last.pt: {val_model_path}")
    else:
        val_model_path = MODEL_PATH
        print(
            f"\n[WARN] No trained weights found. Using initial model: {val_model_path}")

    # -----------------------------------------------------
    # ВАЛИДАЦИЯ
    # -----------------------------------------------------
    val_model = YOLO(str(val_model_path))

    val_metrics = val_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,     # 0 на Windows — нужно везде
    )

    # Безопасное извлечение метрик (map75 есть не во всех версиях ultralytics)
    map50 = float(getattr(val_metrics.box, "map50",  0.0))
    map50_95 = float(getattr(val_metrics.box, "map",    0.0))
    map75 = float(getattr(val_metrics.box, "map75",  0.0))

    # -----------------------------------------------------
    # СВОДКА
    # -----------------------------------------------------
    summary = {
        "project_root":       str(PROJECT_ROOT),
        "dataset_root":       str(DATASET_ROOT),
        "data_yaml":          str(DATA_YAML),
        "initial_model_path": str(MODEL_PATH),
        "save_dir":           str(save_dir),
        "val_model_used":     str(val_model_path),
        "best_model_path":    str(best_model_path) if best_model_path.exists() else None,
        "last_model_path":    str(last_model_path) if last_model_path.exists() else None,
        "device":             DEVICE,
        "workers":            WORKERS,
        "epochs_configured":  EPOCHS,
        "train_images":       len(train_images),
        "val_images":         len(val_images),
        "train_labels":       len(train_labels),
        "val_labels":         len(val_labels),
        "map50":              map50,
        "map50_95":           map50_95,
        "map75":              map75,
    }

    summary_path = save_dir / "training_summary.json"
    summary_path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nBest weights :",
          best_model_path if best_model_path.exists() else "— not found")
    print("Last weights  :", last_model_path if last_model_path.exists()
          else "— not found")
    print("Summary saved :", summary_path)


# =========================================================
# WINDOWS SAFE ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    main()
