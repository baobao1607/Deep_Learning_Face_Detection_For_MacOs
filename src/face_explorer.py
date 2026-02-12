import os
import json
import cv2
import time
from multiprocessing import get_context, cpu_count
from typing import Dict, List, Tuple, Optional

_YUNET = None
_YUNET_CONF = 0.6
_YUNET_INPUT = (320, 320)  # (w, h)

def _init_worker_yunet(model_path: str, conf_thresh: float, input_size: Tuple[int, int]):
    global _YUNET, _YUNET_CONF, _YUNET_INPUT
    _YUNET_CONF = float(conf_thresh)
    _YUNET_INPUT = (int(input_size[0]), int(input_size[1]))

    _YUNET = cv2.FaceDetectorYN.create(
        model_path,
        "",              
        _YUNET_INPUT,    
        _YUNET_CONF,     
        0.3,             
        5000             
    )

def _has_face_yunet_batch(batch_paths: List[str]) -> List[Tuple[str, bool]]:
    global _YUNET, _YUNET_INPUT

    out: List[Tuple[str, bool]] = []
    w_in, h_in = _YUNET_INPUT

    for full_image_path in batch_paths:
        try:
            img = cv2.imread(full_image_path, cv2.IMREAD_REDUCED_COLOR_2)
            if img is None:
                out.append((full_image_path, False))
                continue

            resized = cv2.resize(img, (w_in, h_in), interpolation=cv2.INTER_AREA)

            _, faces = _YUNET.detect(resized)

            has_face = (faces is not None) and (len(faces) > 0)
            out.append((full_image_path, bool(has_face)))

        except Exception:
            out.append((full_image_path, False))

    return out


class FaceExplorer:

    def __init__(
        self,
        raw_dir: str,
        metadata_dir: str,
        yunet_model_path: str = "./models/face_detection_yunet_2023mar.onnx",
        conf_thresh: float = 0.6,
        processes: Optional[int] = None,
        mp_start_method: str = "spawn",   
        progress_every: int = 2000,
        batch_size: int = 32,             
        input_size: Tuple[int, int] = (320, 320),
    ):
        self.raw_dir = raw_dir
        self.metadata_dir = metadata_dir
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.raw_index_file = os.path.join(self.metadata_dir, "raw_index.json")
        self.face_cache_file = os.path.join(self.metadata_dir, "face_check.json")
        self.explore_done_file = os.path.join(self.metadata_dir, "explore.done")

        self.yunet_model_path = yunet_model_path
        self.conf_thresh = conf_thresh
        self.input_size = input_size

        self.processes = processes or max(1, min(cpu_count(), 8))
        self.mp_start_method = mp_start_method
        self.progress_every = max(1, int(progress_every))
        self.batch_size = max(1, int(batch_size))

        self.validate_model_files()

    def validate_model_files(self):
        if not os.path.exists(self.yunet_model_path):
            raise FileNotFoundError(
                f"Missing YuNet model file: {self.yunet_model_path}\n"
                f"Download face_detection_yunet_2022mar.onnx and place it there."
            )

    def load_raw_index(self) -> List[str]:
        if not os.path.exists(self.raw_index_file):
            raise FileNotFoundError(f"Missing JSON file: {self.raw_index_file}")

        with open(self.raw_index_file, "r") as f:
            data = json.load(f)

        if "images" not in data or not isinstance(data["images"], list):
            raise ValueError("raw_index.json missing key 'images' list")

        return data["images"]

    def load_face_cache(self) -> Dict[str, bool]:
        if not os.path.exists(self.face_cache_file):
            return {}
        with open(self.face_cache_file, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {k: bool(v) for k, v in data.items()}

    def save_face_cache(self, cache: Dict[str, bool]) -> None:
        with open(self.face_cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    def _touch_done(self) -> None:
        with open(self.explore_done_file, "w") as f:
            f.write("done\n")

    def _summary(self, rel_paths: List[str], cache: Dict[str, bool], newly_checked: int, already_checked: int):
        total = len(rel_paths)
        no_face = [p for p in rel_paths if cache.get(p, False) is False]
        has_face = total - len(no_face)

        return {
            "total_images": total,
            "has_face": has_face,
            "no_face": len(no_face),
            "newly_checked": newly_checked,
            "already_checked": already_checked,
            "no_face_paths_preview": no_face[:25],
        }

    def explore_faces(self):
        rel_paths = self.load_raw_index()
        cache = self.load_face_cache()

        unchecked_rel = [p for p in rel_paths if p not in cache]
        already_checked = len(rel_paths) - len(unchecked_rel)

        if not unchecked_rel:
            self._touch_done()
            return self._summary(rel_paths, cache, newly_checked=0, already_checked=already_checked)

        unchecked_abs = [os.path.join(self.raw_dir, p) for p in unchecked_rel]

        batches: List[List[str]] = [
            unchecked_abs[i:i + self.batch_size]
            for i in range(0, len(unchecked_abs), self.batch_size)
        ]

        ctx = get_context(self.mp_start_method)
        start_time = time.time()
        processed = 0

        with ctx.Pool(
            processes=self.processes,
            initializer=_init_worker_yunet,
            initargs=(self.yunet_model_path, self.conf_thresh, self.input_size),
        ) as pool:


            for batch_results in pool.imap_unordered(_has_face_yunet_batch, batches, chunksize=1):
                for abs_path, has_face in batch_results:
                    rel_path = os.path.relpath(abs_path, self.raw_dir)
                    cache[rel_path] = bool(has_face)
                    processed += 1

                    if processed % self.progress_every == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        remaining = len(unchecked_abs) - processed
                        eta = remaining / rate if rate > 0 else 0.0
                        print(
                            f"FaceExplorer progress: {processed}/{len(unchecked_abs)} "
                            f"({rate:.2f} img/s, ETA {eta:.1f}s)"
                        )

        self.save_face_cache(cache)
        self._touch_done()
        return self._summary(rel_paths, cache, newly_checked=len(unchecked_rel), already_checked=already_checked)
