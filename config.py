from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    BASE_DIR: Path = BASE_DIR

    # =====================================================
    # COMMON PROJECT DIRS
    # =====================================================
    DATA_DIR: Path = BASE_DIR / "data"
    VIDEO_DIR: Path = BASE_DIR / "video_price"
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    LOGS_DIR: Path = BASE_DIR / "logs"
    WORK_DIR: Path = BASE_DIR / "work_price_tags"

    GROUNDING_DINO_DIR: Path = BASE_DIR / "grounding-dino-base"
    YOLO_RUNS_DIR: Path = BASE_DIR / "runs_yolo12_640"

    INFERENCE_OUTPUT_DIR: Path = OUTPUT_DIR / "inference"
    TRACK_OUTPUT_DIR: Path = OUTPUT_DIR / "tracker"
    CROPS_DIR: Path = OUTPUT_DIR / "crops"

    # =====================================================
    # COMMON FILE EXTENSIONS / GLOBAL PARAMS
    # =====================================================
    VIDEO_EXTS: Tuple[str, ...] = (
        ".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v"
    )
    IMAGE_EXTS: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".webp"
    )

    RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"

    # =====================================================
    # COMMON LABELS / COLORS
    # =====================================================
    CLASS_ID: int = 0
    CLASS_NAME: str = "price_tag"

    BOX_COLOR: tuple = (0, 60, 255)       # BGR
    TEXT_COLOR: tuple = (255, 255, 255)   # BGR
    BG_COLOR: tuple = (0, 60, 255)        # BGR

    # =====================================================
    # DINO / AUTO-LABELING
    # =====================================================
    TEXT_PROMPT: str = (
        "shelf price tag. price label. retail price tag. barcode label."
    )
    DINO_BOX_THRESHOLD: float = 0.28
    DINO_TEXT_THRESHOLD: float = 0.20

    FRAME_STEP: int = 10
    MIN_BRIGHTNESS: int = 35
    MIN_LAPLACIAN_VAR: int = 60
    VAL_RATIO: float = 0.20

    ROTATE_180_BEFORE_FLIP: bool = False
    HORIZONTAL_FLIP: bool = True
    SAVE_VIS: bool = True

    MIN_BOX_WIDTH_PX: int = 20
    MIN_BOX_HEIGHT_PX: int = 12

    MIN_BOX_WIDTH_NORM: float = 0.015
    MIN_BOX_HEIGHT_NORM: float = 0.012
    MAX_BOX_WIDTH_NORM: float = 0.28
    MAX_BOX_HEIGHT_NORM: float = 0.22

    MIN_BOX_AREA_NORM: float = 0.0002
    MAX_BOX_AREA_NORM: float = 0.06

    MIN_ASPECT_RATIO: float = 0.6
    MAX_ASPECT_RATIO: float = 6.5

    CONTAINMENT_THRESHOLD: float = 0.90
    NMS_IOU_THRESHOLD: float = 0.35
    MAX_EDGE_SHARE: float = 0.32

    # =====================================================
    # YOLO TRAIN
    # =====================================================
    TRAIN_MODEL_FILENAME: str = "yolo12n.pt"
    TRAIN_EXPERIMENT_NAME: str = "price_tags_yolo12n_img640_cleanv1"

    TRAIN_EPOCHS: int = 150
    TRAIN_IMGSZ: int = 640
    TRAIN_BATCH: int = 4
    TRAIN_WORKERS: int = 0
    TRAIN_PATIENCE: int = 35
    TRAIN_SAVE_PERIOD: int = 5

    TRAIN_DEGREES: float = 3.0
    TRAIN_TRANSLATE: float = 0.08
    TRAIN_SCALE: float = 0.20
    TRAIN_SHEAR: float = 1.0
    TRAIN_PERSPECTIVE: float = 0.0005
    TRAIN_FLIPLR: float = 0.5
    TRAIN_FLIPUD: float = 0.0
    TRAIN_HSV_H: float = 0.010
    TRAIN_HSV_S: float = 0.35
    TRAIN_HSV_V: float = 0.25
    TRAIN_MOSAIC: float = 0.15
    TRAIN_MIXUP: float = 0.0
    TRAIN_COPY_PASTE: float = 0.0

    # =====================================================
    # YOLO INFERENCE
    # =====================================================
    INFERENCE_VIDEO_PATH: Path = BASE_DIR / "26_2-10.mp4"
    INFERENCE_MODEL_PATH: Path = (
        BASE_DIR / "runs_yolo12_640" /
        "price_tags_yolo12s_img640_cleanv1-2" / "weights" / "best.pt"
    )

    INFER_CONF_THRESH: float = 0.40
    INFER_IOU_THRESH: float = 0.45
    INFER_IMGSZ: int = 640
    INFER_SAVE_VIDEO: bool = True
    INFER_SHOW_WINDOW: bool = True
    INFER_SKIP_FRAMES: int = 0

    # =====================================================
    # CROP EXTRACTION
    # =====================================================
    CROP_MODEL_PATH: Path = (
        BASE_DIR / "runs_yolo12_640" /
        "price_tags_yolo12n_img640_cleanv1" / "weights" / "best.pt"
    )
    CROP_SOURCE_DIR: Path = VIDEO_DIR
    CROP_OUTPUT_DIR: Path = CROPS_DIR

    CROP_CONF_THRESH: float = 0.40
    CROP_IOU_THRESH: float = 0.45
    CROP_IMGSZ: int = 640

    PADDING: int = 4
    MIN_CROP_W: int = 20
    MIN_CROP_H: int = 12
    CROP_FRAME_STEP: int = 10

    # =====================================================
    # TRACKER
    # =====================================================
    TRACK_MODEL_PATH: Path = (
        BASE_DIR / "runs_yolo12_640" /
        "price_tags_yolo12s_img640_cleanv1-2" / "weights" / "best.pt"
    )
    TRACK_VIDEO_PATH: Path = BASE_DIR / "26_2-10.mp4"

    TRACK_CONF_THRESH: float = 0.40
    TRACK_IOU_THRESH: float = 0.45
    TRACK_IMGSZ: int = 640
    TRACK_SAVE_VIDEO: bool = True
    TRACK_SHOW_WINDOW: bool = True
    TRACK_SKIP_FRAMES: int = 0

    # =====================================================
    # CLEANUP AFTER MANUAL REVIEW
    # =====================================================
    CLEANUP_DO_DELETE: bool = False
    CLEANUP_UPDATE_PIPELINE_STATS: bool = False
    CLEANUP_REPORT_FILENAME: str = "cleanup_report.json"

    # =====================================================
    # DERIVED PATHS
    # =====================================================
    @property
    def DATASET_ROOT(self) -> Path:
        return self.WORK_DIR / "dataset"

    @property
    def DATA_YAML(self) -> Path:
        return self.DATASET_ROOT / "data.yaml"

    @property
    def YOLO_WEIGHTS_DIR(self) -> Path:
        return self.BASE_DIR

    @property
    def TRAIN_MODEL_PATH(self) -> Path:
        return self.YOLO_WEIGHTS_DIR / self.TRAIN_MODEL_FILENAME

    @property
    def IMAGES_ALL_DIR(self) -> Path:
        return self.DATASET_ROOT / "images_all"

    @property
    def LABELS_ALL_DIR(self) -> Path:
        return self.DATASET_ROOT / "labels_all"

    @property
    def IMAGES_TRAIN_DIR(self) -> Path:
        return self.DATASET_ROOT / "images" / "train"

    @property
    def IMAGES_VAL_DIR(self) -> Path:
        return self.DATASET_ROOT / "images" / "val"

    @property
    def LABELS_TRAIN_DIR(self) -> Path:
        return self.DATASET_ROOT / "labels" / "train"

    @property
    def LABELS_VAL_DIR(self) -> Path:
        return self.DATASET_ROOT / "labels" / "val"

    @property
    def VIS_DIR(self) -> Path:
        return self.DATASET_ROOT / "vis"

    @property
    def META_DIR(self) -> Path:
        return self.DATASET_ROOT / "meta"

    @property
    def DINO_VIS_DIR(self) -> Path:
        return self.VIS_DIR

    @property
    def DINO_META_DIR(self) -> Path:
        return self.META_DIR

    @property
    def PIPELINE_STATS_PATH(self) -> Path:
        return self.DATASET_ROOT / "pipeline_stats.json"

    @property
    def CLEANUP_REPORT_PATH(self) -> Path:
        return self.DATASET_ROOT / self.CLEANUP_REPORT_FILENAME


CFG = Config()
