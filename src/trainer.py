import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # hides all TF C++ logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'    # prevents grappler pruner warnings
os.environ['GRAPHLITE_DISABLE'] = '1'        # additional grappler suppression

import tensorflow as tf


class Trainer:
    def __init__(self, model, ckpt_dir="./checkpoints", log_dir="./logs", model_name="default"):
        self.model = model.model
        safe_model_name = model_name.lstrip("/\\")
        self.ckpt_dir = os.path.join(ckpt_dir, safe_model_name)
        self.log_dir = os.path.join(log_dir, safe_model_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_name = safe_model_name

        self.ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.ckpt_dir, "best_model.keras"),
            save_weights_only=False,
            save_freq='epoch',
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        )

        self.last_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.ckpt_dir, "last_model.keras"),
            save_weights_only=False,
            save_best_only=False,
            save_freq="epoch",
            verbose=1,
        )


        self.tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False
        )

        self.es_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose = 1
        )

        self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = "val_loss",
            factor = 0.5,
            patience = 2,
            min_lr = 1e-6,
        )

    def _load_checkpoint(self, checkpoint_path):
        self.model = tf.keras.models.load_model(checkpoint_path)

    def train(self, train_ds, val_ds, epochs=20, class_weight=None):
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[self.ckpt_cb, self.es_cb,self.tb_cb, self.last_ckpt_cb, self.lr_scheduler],
            class_weight=class_weight,
            verbose = 1
        )
        return history


    def resume(self, checkpoint_path, train_ds, val_ds, epochs=10):
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"TensorBoard log_dir: {self.log_dir}")
        self._load_checkpoint(checkpoint_path)


        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[self.ckpt_cb, self.es_cb, self.tb_cb, self.last_ckpt_cb, self.lr_scheduler]
        )


    def restore_model(self, checkpoint_path):
        self._load_checkpoint(checkpoint_path)
        return self.model
    
    def evaluate(self, test_ds):
        return self.model.evaluate(test_ds, verbose=2)


    def predict_one(self, img_tensor):
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        prob = self.model.predict(img_tensor)[0][0]
        return prob
