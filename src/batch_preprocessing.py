import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'    
os.environ['GRAPHLITE_DISABLE'] = '1'        

import tensorflow as tf
import json
import random


class BatchPreprocessor:

    def __init__(
            self,
            metadata_dir,
            dataset_dir,
            image_size ,
            batch_size,
            shuffle_buffer,
            augment: bool
    ):
        self.metadata_dir = metadata_dir
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.augment = augment
        self.normalizer = self.build_normalizer()
        self.augmenter = self.build_augmenter()
        self.batch_done_file = os.path.join(self.metadata_dir, "batch.done")
        self.split_index = os.path.join(self.metadata_dir,"split_index.json")
        self.seed = 42

    def build_normalizer(self):
        return tf.keras.layers.Rescaling(1./255)
    
    def build_augmenter(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1)
            ]
        )
    
    def get_labels(self, relative_path):
        parts = relative_path.split(os.sep)
        gen_folder = parts[0].lower()

        if gen_folder == "ffhq":
            return 0
        else:
            return 1
        
        
    
    def load_split_paths(self, split):
        with open(self.split_index, "r") as f:
            index = json.load(f)

        rel_paths = index[split]
        return rel_paths
    
    def file_level_shuffle(self, paths, labels):
        rng = random.Random(self.seed)
        combined = list(zip(paths, labels))
        rng.shuffle(combined)
        if not combined:
            return [], []
        paths, labels = zip(*combined)
        return list(paths), list(labels)

    def load_image(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels = 3)
        img = tf.image.resize(img, self.image_size)
        return img


    def preprocesses(self, path, label):
        img = self.load_image(path)
        img = self.normalizer(img)
        return img, label

    def build_pipeline(self, split, shuffle):
        data_dir = os.path.join(self.dataset_dir, split)

        rel_paths = self.load_split_paths(split)
        labels = [self.get_labels(p) for p in rel_paths]
        full_paths = [
            os.path.join(data_dir, str(label), os.path.basename(rel_path))
            for rel_path, label in zip(rel_paths, labels)
        ]

        if split == "train":
            full_paths, labels = self.file_level_shuffle(full_paths, labels)

        ds = tf.data.Dataset.from_tensor_slices((full_paths, labels))
        ds = ds.shuffle(len(full_paths), reshuffle_each_iteration=True)


        if split != "test":
            ds = ds.map(
                lambda p, y: self.preprocesses(p, y),
                num_parallel_calls= tf.data.AUTOTUNE
            )
        else:
            ds = ds.map(
                lambda p, y: self.preprocesses(p, y),
                num_parallel_calls= 1
            )

        if shuffle and self.augment:
            ds = ds.map(
                lambda x,y: (self.augmenter(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    
    def build_train_ds(self):
        return self.build_pipeline("train",True)
    
    def build_valid_ds(self):
        return self.build_pipeline("valid",False)
    
    def build_test_ds(self):
        return self.build_pipeline("test",False)

    def build_all(self):
        train = self.build_train_ds()
        valid = self.build_valid_ds()
        test = self.build_test_ds()
        self.summary()
        return train, valid, test

    
    def summary(self):
        print(f"Dataset directory : {self.dataset_dir}")
        print(f"Image size        : {self.image_size}")
        print(f"Batch size        : {self.batch_size}")
        print(f"Shuffle buffer    : {self.shuffle_buffer}")
        print(f"Augmentation      : {self.augment}")

    

        
