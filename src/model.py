import tensorflow as tf

class BaseModel:
    def __init__(self, input_shape=(214, 214, 3), lr=3e-5):
        self.input_shape = input_shape
        self.lr = lr
        self.model = self._build_model()
        self.compile()

    def _build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs, name="BaselineModel")
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