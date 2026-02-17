import os
import numpy as np


class DatasetManager:
    def __init__(
            self,
            metadata_dir,
            raw_data_indexer,
            face_explorer,
            dataset_splitter,
            batcher
    ):
        self.metadata_dir = metadata_dir
        self.raw_indexer = raw_data_indexer
        self.face_explorer = face_explorer
        self.splitter = dataset_splitter
        self.batcher = batcher

        self.face_done = os.path.join(self.metadata_dir,"explore.done")
        self.split_done = os.path.join(self.metadata_dir,"split.done")

        self.hash_file = os.path.join(self.metadata_dir,"raw_hash.txt")

    def _is_done(self, marker):
        return os.path.exists(marker)
    
    def _touch_done(self,marker):
        with open(marker, "w") as f:
            f.write("done\n")

    def _invalidate(self, marker: str):
        if os.path.exists(marker):
            os.remove(marker)
    
    def load_old_hash(self):
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r") as f:
                return f.read().strip()
    
    def run_pipeline(self):
        print(f"Starting .....")
        print("EXPLORING.....")
        old_hash = self.load_old_hash()
        index_result = self.raw_indexer.scan_and_detect()
        new_hash = index_result["hash"].strip()
        
        if (new_hash != old_hash):
            print("INVALIDATING MARKER\n")
            self._invalidate(self.face_done)
            self._invalidate(self.split_done)
        else:
            print("DATA IS INTEGRITY\n")
        
        if not self._is_done(self.face_done):
            print("EXPLORING FACE...")
            explore_summary = self.face_explorer.explore_faces()
            print(explore_summary)
            print("\n")
        else:
            print("FaceExplorer already complete. Skipping.\n")

        if not self._is_done(self.split_done):
            print("SPLITTING......")
            split_summary = self.splitter.split_datasource()
            print(split_summary)
            print("\n")
        else:
            print("DatasetSplitter already done. Skipping.\n")
        print("BatchPreprocessing.......")
        train_ds, valid_ds, test_ds = self.batcher.build_all()
        print("\n")

        return train_ds, valid_ds, test_ds
    
    def _count_labels_from_split_index(self):
        split_index_path = os.path.join(self.metadata_dir, "split_index.json")
        if not os.path.exists(split_index_path):
            return None

        import json
        with open(split_index_path, "r") as f:
            split_index = json.load(f)

        train_items = split_index.get("train", [])
        if not train_items:
            return None

        neg = 0
        pos = 0
        for rel_path in train_items:
            folder = rel_path.split(os.sep)[0].lower()
            if folder == "ffhq":
                neg += 1
            else:
                pos += 1

        return neg, pos

    def compute_class_weights(self, train_ds):
        counts = self._count_labels_from_split_index()
        if counts is not None:
            neg, pos = counts
        else:
            y = []
            for _, labels in train_ds:
                y.append(labels.numpy())
            y = np.concatenate(y)
            neg = (y == 0).sum()
            pos = (y == 1).sum()

        total = neg + pos

        w0 = total / (2.0 * neg)
        w1 = total / (2.0 * pos)

        print(f"Class 0: {neg}, Class 1: {pos}")
        print(f"Weight 0: {w0:.4f}, Weight 1: {w1:.4f}")

        return {0: w0, 1: w1}
