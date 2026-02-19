import argparse
from src.batch_preprocessing import BatchPreprocessor
from src.model import BaseModel
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
    model_dir = "./checkpoints"
    test_root_dir = "./data/dataset/test"


    batcher = BatchPreprocessor(
        metadata_dir=metadata_dir,
        dataset_dir=dataset_dir,  
        image_size=(214, 214),
        batch_size=48,
        shuffle_buffer=None,
        augment=True
    )

    model = BaseModel(lr=1e-4)
    model.summary()

    trainer = Trainer(model=model, model_name=model_name)
    checkpoint_path = os.path.join(model_dir,model_name,"best_model.keras")
    model = trainer.restore_model(checkpoint_path)
    __,___,test_ds = batcher.build_all()

    print(f"EVALUATNG STARTING...")
    print(f"Combined test set")
    trainer.evaluate(test_ds)
    print(f"-"*40)

    print(f"Generator seperated test set")
    for subfolder in os.listdir(test_root_dir):
        print(f"Evaluating {subfolder}")
        sub_path = os.path.join(test_root_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue
        test_ds = batcher.build_pipeline_for_testing(subfolder)
        result = trainer.evaluate(test_ds)

        precision = result[2]
        recall = result[3]

        if precision + recall > 0:
            F1_score = 2 * precision * recall / (precision + recall)
        else:
            F1_score = 0.0

        print(f"F1 Score: {F1_score:.4f}")
        print(f"-"*40)
        



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
