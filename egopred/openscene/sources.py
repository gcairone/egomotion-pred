"""Remote locations of OpenScene / NAVSIM files, pinned to fixed revisions."""

# Hugging Face dataset OpenDriveLab/OpenScene (CC BY-NC-SA 4.0).
HF_REVISION = "a76f840b65e972bc45e56c2adced897498e9a026"
HF_BASE = f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{HF_REVISION}"

# NAVSIM devkit, for the official scene filters (navtrain / navtest token lists).
NAVSIM_REVISION = "0a380a9063d7162ec93d0f51e9990ebac585f720"
SCENE_FILTER_URL = (
    f"https://raw.githubusercontent.com/autonomousvision/navsim/{NAVSIM_REVISION}"
    "/navsim/planning/script/config/common/train_test_split/scene_filter/{name}.yaml"
)

NUM_CAMERA_ARCHIVES = {"mini": 32, "trainval": 200, "test": 32}
NUM_NAVTRAIN_ARCHIVES = 32


def metadata_url(split: str) -> str:
    return f"{HF_BASE}/openscene-v1.1/openscene_metadata_{split}.tgz"


def camera_archive(split: str, i: int) -> tuple[str, str]:
    """(archive name, url) of the i-th all-camera archive of an OpenScene split."""
    name = f"openscene_sensor_{split}_camera_{i}"
    return name, f"{HF_BASE}/openscene-v1.1/openscene_sensor_{split}_camera/{name}.tgz"


def navtrain_archive(kind: str, i: int) -> tuple[str, str]:
    """(archive name, url) of a NAVSIM navtrain sensor archive; kind is current|history."""
    if kind not in ("current", "history"):
        raise ValueError(kind)
    name = f"navtrain_{kind}_{i}"
    return name, f"{HF_BASE}/navsim/{name}.tgz"
