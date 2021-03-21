from datetime import datetime

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.python.keras.layers import BatchNormalization, Dense, Flatten, Concatenate

from chess_dataset import ChessDataset


def define_model(input_shape, player_input_shape, filters=4):
    input_ = Input(shape=input_shape)
    player_input = Input(player_input_shape)
    p = Dense(8)(player_input)
    padding = "same"

    # x = Conv2D(filters, 2, 1, padding=padding, activation="relu")(input_)
    # x = BatchNormalization()(x)
    # x = Conv2D(filters, 2, 1, padding=padding, activation="relu")(x)
    # x = BatchNormalization()(x)
    #
    # x2 = Conv2D(filters, 4, 1, padding=padding, activation="relu")(input_)
    # x2 = BatchNormalization()(x2)
    # x2 = Conv2D(filters, 4, 1, padding=padding, activation="relu")(x2)
    # x2 = BatchNormalization()(x2)
    #
    # x3 = Conv2D(filters, 6, 1, padding=padding, activation="relu")(input_)
    # x3 = BatchNormalization()(x3)
    # x3 = Conv2D(filters, 6, 1, padding=padding, activation="relu")(x3)
    # x3 = BatchNormalization()(x3)
    #
    # x4 = Conv2D(filters, 8, 1, padding=padding, activation="relu")(input_)
    # x4 = BatchNormalization()(x4)
    # x4 = Conv2D(filters, 8, 1, padding=padding, activation="relu")(x4)
    # x4 = BatchNormalization()(x4)

    # x = Concatenate()([x, x2, x3, x4])

    x = Conv2D(filters, 3, 1, padding=padding, activation="relu")(input_)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, 1, padding=padding, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Concatenate()([input_, x])

    x2 = Conv2D(filters, 3, 1, padding=padding, activation="relu")(x)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(filters, 3, 1, padding=padding, activation="relu")(x2)
    x2 = BatchNormalization()(x2)

    x3 = Concatenate()([x, x2])

    # x3 = Conv2D(2 * filters, 5, 1, padding="valid", activation="relu")(x2)
    # x3 = BatchNormalization()(x3)
    # x3 = Conv2D(filters, 3, 1, padding=padding, activation="relu")(x2)
    # x3 = BatchNormalization()(x3)
    # x3 = Conv2D(filters, 3, 1, padding=padding, activation="relu")(x3)
    # x2 = BatchNormalization()(x3)
    #
    # x3 = Concatenate()([x2, x3])

    x3 = Flatten()(x3)
    x3 = Concatenate()([x3, p])
    output = Dense(64 * 64, activation="softmax")(x3)

    model = k.Model(inputs=[input_, player_input], outputs=output)
    model.summary()
    model.compile(
        loss=k.losses.categorical_crossentropy,
        optimizer=k.optimizers.Adam(lr=0.001),
        metrics=["accuracy"])
    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )
    return model


def define_callbacks():
    reduce_lr = k.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                                              patience=2, min_lr=0.0001)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=2),
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='models/model_{epoch:02d}_{val_accuracy:.2f}.h5'),
    return [reduce_lr, early_stop, model_checkpoint]


if __name__ == '__main__':
    input_shape = (8, 8, 8)
    epochs = 30
    batch_size = 1024
    model = define_model(input_shape, (8,))
    # model = tf.keras.models.load_model("models/epochs_30_batch_size_512_2021-03-14_18:58:16.660242.h5")

    model.fit(
        ChessDataset(batch_size=batch_size),
        epochs=epochs,
        validation_data=ChessDataset(batch_size=batch_size, validation=True),
        workers=4,
        shuffle=True,
        callbacks=define_callbacks()
    )
    model.save(f"models/epochs_{epochs}_batch_size_{batch_size}_{str(datetime.now()).replace(' ', '_')}.h5")
