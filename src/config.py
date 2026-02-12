from pathlib import Path

# Automatically detect project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Default local data folder (used if no --data_root is passed)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def get_paths(data_root: str = None):
    """
    Centralized path manager for the entire project.
    Works locally and in Colab.

    Args:
        data_root (str, optional):
            If provided, overrides default data directory.
            Example for Colab:
            "/content/drive/MyDrive/dataset"
    """

    # Data root (dataset location)
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT

    return {
        # Core roots
        "project_root": PROJECT_ROOT,
        "data_root": root,

        # Data structure
        "raw_dir": root / "data_source",
        "dataset_dir": root / "dataset",
        "metadata_dir": root / "dataset" / "metadata",

        # Model files
        "model_dir": PROJECT_ROOT / "models",
        "yunet_model": PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx",
    }
