import tensorflow
import tensorflow.keras as keras

def ImageClassifier(inputShape=[28,28,1], numClasses=10):
    model = keras.models.Sequential([
        keras.layers.Input(inputShape),
        keras.layers.Conv2D(64, 3, strides=[3, 3], activation='relu'),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(24, activation=keras.activations.relu),
        keras.layers.Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])
    return model