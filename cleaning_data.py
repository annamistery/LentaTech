# Скрипт удаления нерелевантных изображений после ручной проверки

import json
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
ROOT = Path(r"E:\Data Science\Хакатон_ценники\work_price_tags\dataset").resolve()

VIS_DIR = ROOT / "vis"
IMAGES_ALL_DIR = ROOT / "images_all"
LABELS_ALL_DIR = ROOT / "labels_all"
META_DIR = ROOT / "meta"

IMAGES_TRAIN_DIR = ROOT / "images" / "train"
IMAGES_VAL_DIR = ROOT / "images" / "val"
LABELS_TRAIN_DIR = ROOT / "labels" / "train"
LABELS_VAL_DIR = ROOT / "labels" / "val"

PIPELINE_STATS_PATH = ROOT / "pipeline_stats.json"

DO_DELETE = True          # True = удалять, False = только показать что будет удалено
# True = переписать pipeline_stats.json краткой сводкой
UPDATE_PIPELINE_STATS = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_EXTS = {".txt"}
META_EXTS = {".json"}


# =========================================================
# HELPERS
# =========================================================
def vis_to_key(path: Path) -> str:
    name = path.stem  # 14931145132657_frame_000000_vis
    if name.endswith("_vis"):
        return name[:-4]
    return name


def file_to_key(path: Path) -> str:
    return path.stem


def collect_vis_keys(vis_dir: Path) -> set[str]:
    keys = set()
    for p in vis_dir.iterdir():
        if p.is_file():
            keys.add(vis_to_key(p))
    return keys


def collect_orphans(target_dir: Path, valid_keys: set[str], allowed_exts: set[str]):
    orphans = []
    if not target_dir.exists():
        return orphans

    for p in target_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed_exts:
            continue
        key = file_to_key(p)
        if key not in valid_keys:
            orphans.append(p)
    return orphans


def delete_files(files):
    deleted = 0
    for p in files:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"[WARN] Cannot delete {p}: {e}")
    return deleted


# =========================================================
# MAIN
# =========================================================
def main():
    if not VIS_DIR.exists():
        raise FileNotFoundError(f"VIS folder not found: {VIS_DIR}")

    valid_keys = collect_vis_keys(VIS_DIR)
    print(f"Valid keys from vis: {len(valid_keys)}")

    targets = [
        ("images_all", IMAGES_ALL_DIR, IMAGE_EXTS),
        ("labels_all", LABELS_ALL_DIR, LABEL_EXTS),
        ("meta", META_DIR, META_EXTS),
        ("images/train", IMAGES_TRAIN_DIR, IMAGE_EXTS),
        ("images/val", IMAGES_VAL_DIR, IMAGE_EXTS),
        ("labels/train", LABELS_TRAIN_DIR, LABEL_EXTS),
        ("labels/val", LABELS_VAL_DIR, LABEL_EXTS),
    ]

    report = {}
    total_to_delete = 0
    all_orphans = []

    for name, folder, exts in targets:
        orphans = collect_orphans(folder, valid_keys, exts)
        report[name] = {
            "folder": str(folder),
            "orphans_count": len(orphans),
            # только первые 20 в отчёт
            "orphans": [str(p) for p in orphans[:20]],
        }
        total_to_delete += len(orphans)
        all_orphans.extend(orphans)

    print("\n=== DRY REPORT ===")
    for name, info in report.items():
        print(f"{name:15s}: {info['orphans_count']} files to delete")

    print(f"\nTotal files to delete: {total_to_delete}")

    if not DO_DELETE:
        print("\nDO_DELETE = False, nothing removed.")
        return

    deleted = delete_files(all_orphans)
    print(f"\nDeleted files: {deleted}")

    # Опционально: обновить pipeline_stats.json
    if UPDATE_PIPELINE_STATS:
        stats = {
            "cleanup_based_on_vis": True,
            "vis_keys_count": len(valid_keys),
            "deleted_files_total": deleted,
            "deleted_breakdown": {
                name: report[name]["orphans_count"] for name, _, _ in targets
            }
        }
        PIPELINE_STATS_PATH.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"Updated pipeline_stats.json: {PIPELINE_STATS_PATH}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
