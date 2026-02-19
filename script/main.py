import argparse
from src.batch_preprocessing import BatchPreprocessor
from src.model_improve import BaseModel
from src.trainer import Trainer
import tensorflow as tf
import numpy as np
import random
import os

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



def main(model_name):
    dataset_dir = "./data/dataset"
    metadata_dir = "./data/dataset/metadata"


    batcher = BatchPreprocessor(
        metadata_dir=metadata_dir,
        dataset_dir=dataset_dir,  
        image_size=(214, 214),
        batch_size=48,
        shuffle_buffer=None,
        augment=True
    )

    train_ds, valid_ds, test_ds = batcher.build_all()

    print("Pipeline finished\n")
    print("START TRAINING")
    

    model = BaseModel(lr=1e-4)
    model.summary()

    trainer = Trainer(model=model, model_name=model_name)
    trainer.train(
        train_ds = train_ds,
        val_ds= valid_ds,
        epochs = 20
    )


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


    ''' dataset_dir = "./data/dataset"
    train_path = os.path.join(dataset_dir,"test")
    total_count  = 0
    for folder in os.listdir(train_path):
        folder_path = os.path.join(train_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # count only files (and optionally only images)
        count = sum(
            1 for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        )
        total_count += count

        print(f"{folder}: {count}")
    print(f"total count {total_count}")'''
