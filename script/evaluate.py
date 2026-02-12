import argparse

from src.raw_data_indexer import RawDataIndexer
from src.face_explorer import FaceExplorer
from src.dataset_splitter import DatasetSplitter
from src.batch_preprocessing import BatchPreprocessor
from src.dataset_manager import DatasetManager  
from src.model import BaseModel
from src.trainer import Trainer
import os



def main(model_name):
    raw_dir = "./data/data_source"
    dataset_dir = "./data/dataset"
    metadata_dir = "./data/dataset/metadata"


    indexer = RawDataIndexer(raw_dir, metadata_dir)

    explorer = FaceExplorer(
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        conf_thresh=0.6,
        processes=None
    )

    splitter = DatasetSplitter(
        raw_dir=raw_dir,
        dataset_dir=dataset_dir,
        metadata_dir=metadata_dir,
        train_ratio=0.75,
        valid_ratio=0.15,
        test_ratio=0.10
    )

    batcher = BatchPreprocessor(
        metadata_dir=metadata_dir,
        dataset_dir=dataset_dir,  
        image_size=(214, 214),
        batch_size=32,
        shuffle_buffer=None,
        augment=True
    )

    manager = DatasetManager(
        metadata_dir=metadata_dir,
        raw_data_indexer=indexer,
        face_explorer=explorer,
        dataset_splitter=splitter,
        batcher=batcher
    )

    train_ds, valid_ds, test_ds = manager.run_pipeline()
    class_weight = manager.compute_class_weights(train_ds)

    print("Pipeline finished\n")
    print("START TRAINING")
    

    model = BaseModel(lr=1e-3)
    model.summary()

    trainer = Trainer(model=model, model_name=model_name)
    model_path = os.path.join(trainer.ckpt_dir,"best_model.keras")
    model = trainer.restore_model(model_path)

    print("EVALUATING")
    trainer.evaluate(test_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model run name used for checkpoints/logs (e.g., baseline_v1).",
    )
    args = parser.parse_args()
    main(model_name=args.model_name)
