from tensorflow.keras import layers, Model, Sequential

def build_cnn(input_shape):
    model = Sequential([
        layers.Conv2D(16, (7,7), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (7,7), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2DTranspose(32, (7,7), strides=2, activation="relu", padding="same"),
        layers.Conv2DTranspose(16, (7,7), strides=2, activation="relu", padding="same"),
        layers.Conv2D(1, (1,1), activation="sigmoid")
    ])
    return model

def build_unet(input_shape, n_layers=2, n_filters=64):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []

    for i in range(n_layers):
        f = n_filters * (2 ** i)
        x = layers.Conv2D(f, (3,3), activation="relu", padding="same")(x)
        x = layers.Conv2D(f, (3,3), activation="relu", padding="same")(x)
        skips.append(x)
        x = layers.MaxPooling2D((2,2))(x)

    for i in reversed(range(n_layers)):
        f = n_filters * (2 ** i)
        x = layers.Conv2DTranspose(f, (3,3), strides=2, activation="relu")(x)
        x = layers.concatenate([x, skips[i]])
        x = layers.Conv2D(f, (3,3), activation="relu", padding="same")(x)
        x = layers.Conv2D(f, (3,3), activation="relu", padding="same")(x)

    outputs = layers.Conv2D(1, (1,1), activation="sigmoid")(x)
    return Model(inputs, outputs)
