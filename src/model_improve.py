import tensorflow as tf


class BaseModel:
    def __init__(self, input_shape=(214, 214, 3), lr=1e-4):
        self.input_shape = input_shape
        self.lr = lr
        self.model = self._build_model()
        self.compile()


    def residual_block(self, x , filters, downsample=False):
        shortcut = x

        stride = 2 if downsample else 1

        x = tf.keras.layers.Conv2D(filters, 3, strides = stride, padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if downsample or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.Add()([x,shortcut])
        x = tf.keras.layers.ReLU()(x)

        return x

    def _build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x  = tf.keras.layers.ReLU()(x)

        x = self.residual_block(x, 64, downsample=True)
        x = self.residual_block(x, 64)

        x = self.residual_block(x, 128, downsample= True)
        x = self.residual_block(x, 128)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs, name="ImprovedModel")
        return model

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )

    def summary(self):
        return self.model.summary()