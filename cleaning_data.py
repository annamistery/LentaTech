import json
from pathlib import Path
from typing import Dict, List, Set

from config import CFG
from logging_setup import setup_logging


logger = setup_logging("cleanup_after_review")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_EXTS = {".txt"}
META_EXTS = {".json"}


def vis_to_key(path: Path) -> str:
    name = path.stem
    if name.endswith("_vis"):
        return name[:-4]
    return name


def file_to_key(path: Path) -> str:
    return path.stem


def collect_vis_keys(vis_dir: Path) -> Set[str]:
    if not vis_dir.exists():
        raise FileNotFoundError(f"VIS folder not found: {vis_dir}")

    keys = set()
    for p in vis_dir.iterdir():
        if p.is_file():
            keys.add(vis_to_key(p))
    return keys


def collect_orphans(target_dir: Path, valid_keys: Set[str], allowed_exts: Set[str]) -> List[Path]:
    orphans: List[Path] = []

    if not target_dir.exists():
        logger.warning(
            "Target directory does not exist, skipping: %s", target_dir)
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


def delete_files(files: List[Path]) -> int:
    deleted = 0

    for p in files:
        try:
            p.unlink()
            deleted += 1
            logger.info("Deleted: %s", p)
        except Exception:
            logger.exception("Cannot delete file: %s", p)

    return deleted


def build_targets():
    return [
        ("images_all", CFG.IMAGES_ALL_DIR, IMAGE_EXTS),
        ("labels_all", CFG.LABELS_ALL_DIR, LABEL_EXTS),
        ("meta", CFG.META_DIR, META_EXTS),
        ("images/train", CFG.IMAGES_TRAIN_DIR, IMAGE_EXTS),
        ("images/val", CFG.IMAGES_VAL_DIR, IMAGE_EXTS),
        ("labels/train", CFG.LABELS_TRAIN_DIR, LABEL_EXTS),
        ("labels/val", CFG.LABELS_VAL_DIR, LABEL_EXTS),
    ]


def write_cleanup_report(report: Dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Cleanup report saved: %s", output_path)


def update_pipeline_stats(valid_keys_count: int, deleted: int, report: Dict) -> None:
    stats = {
        "cleanup_based_on_vis": True,
        "vis_keys_count": valid_keys_count,
        "deleted_files_total": deleted,
        "deleted_breakdown": {
            name: info["orphans_count"]
            for name, info in report["targets"].items()
        },
    }

    CFG.DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    pipeline_stats_path = CFG.DATASET_ROOT / "pipeline_stats.json"
    pipeline_stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Updated pipeline_stats.json: %s", pipeline_stats_path)


def main() -> None:
    valid_keys = collect_vis_keys(CFG.DINO_VIS_DIR)
    logger.info("Valid keys from vis: %d", len(valid_keys))

    targets = build_targets()

    report = {
        "dataset_root": str(CFG.DATASET_ROOT),
        "vis_dir": str(CFG.DINO_VIS_DIR),
        "do_delete": CFG.CLEANUP_DO_DELETE,
        "update_pipeline_stats": CFG.CLEANUP_UPDATE_PIPELINE_STATS,
        "valid_keys_count": len(valid_keys),
        "targets": {},
    }

    total_to_delete = 0
    all_orphans: List[Path] = []

    for name, folder, exts in targets:
        orphans = collect_orphans(folder, valid_keys, exts)
        report["targets"][name] = {
            "folder": str(folder),
            "orphans_count": len(orphans),
            "orphans": [str(p) for p in orphans[:20]],
        }
        total_to_delete += len(orphans)
        all_orphans.extend(orphans)

    logger.info("=== DRY REPORT ===")
    for name, info in report["targets"].items():
        logger.info("%s: %d files to delete", name, info["orphans_count"])

    logger.info("Total files to delete: %d", total_to_delete)

    print("\n=== DRY REPORT ===")
    for name, info in report["targets"].items():
        print(f"{name:15s}: {info['orphans_count']} files to delete")
    print(f"\nTotal files to delete: {total_to_delete}")

    report_path = CFG.DATASET_ROOT / CFG.CLEANUP_REPORT_FILENAME
    write_cleanup_report(report, report_path)

    if not CFG.CLEANUP_DO_DELETE:
        logger.info("CLEANUP_DO_DELETE = False, nothing removed.")
        print("\nCLEANUP_DO_DELETE = False, nothing removed.")
        return

    deleted = delete_files(all_orphans)
    logger.info("Deleted files: %d", deleted)
    print(f"\nDeleted files: {deleted}")

    if CFG.CLEANUP_UPDATE_PIPELINE_STATS:
        update_pipeline_stats(
            valid_keys_count=len(valid_keys),
            deleted=deleted,
            report=report,
        )

    logger.info("=== DONE ===")
    print("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in cleanup_after_review.py")
        raise
