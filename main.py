import os
import time
from datetime import datetime
from pathlib import Path

import fire
import subprocess as sp
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from chess_dataset import ChessDataset
from pgn_to_array_converter import download_pgn, PgnToArrayConverter
from train_chess_cnn_keras import define_model, define_callbacks


def get_data(download_url, rating=2600):
    # download the pgn
    pgn_file = download_pgn(download_url)
    print(f"\nRating threshold set to: {rating}")
    data_converter = PgnToArrayConverter(pgn_file, rating=rating)
    print("\nConverting PGN to numpy arrays")
    data_converter.pgn_to_arrays()


def train_cnn(model_save_name=None, epochs=20, batch_size=256):
    print(f"Beginning training of Chess CNN of selected games for {epochs} epochs")
    input_shape = (8, 8, 8)
    model = define_model(input_shape, (8,))

    model.fit(
        ChessDataset(batch_size=batch_size),
        epochs=epochs,
        validation_data=ChessDataset(batch_size=batch_size, validation=True),
        workers=4,
        shuffle=True,
        callbacks=define_callbacks()
    )
    model.save(model_save_name, overwrite=True)
    # Tensorflow currently hordes the GPU memory after training
    # print("Clearing GPU Memory")
    # tf.config.set_visible_devices([], 'GPU')
    # tf.keras.backend.clear_session()


# def play_cnn(model_path):
#     print("\nRunning Basic Chess GUI")
#     Path("model_name.txt").open("w").write(model_path)
#     cmd = "python kivy_gui.py"
#     time.sleep(5)
#     sp.run(cmd.split())


def download_train_play_chess_cnn(download_url: str = "https://database.nikonoel.fr/lichess_elite_2021-01.zip",
                                  rating: int = 2600, model_name: str = "chess_model.h5", epochs: int = 20):
    """
    The complete pipeline of downloading PGN Zip from Lichess (https://database.nikonoel.fr),
    Training a CNN using Keras (https://keras.io) and then Playing the Engine using Kivy (https://kivy.org).

    Note: Currently need to run "python kivy_gui.py" after this command, as Tensorflow doesn't release the GPU memory
    after training. Likely a bug.
    :param rating: Rating threshold for games to be trained on
    :param model_name: Model save name
    :param epochs: Number of epochs to train the Chess CNN on
    :param download_url: Path to Lichess PGN Zip, E.g. https://database.nikonoel.fr/lichess_elite_2021-01.zip".
    Otherwise can modify the download_pgn function yourself
    """
    get_data(download_url, rating)
    train_cnn(model_name, epochs)
    # commented out due to Tensorflow bug
    # play_cnn(model_name)


if __name__ == '__main__':
    fire.Fire(download_train_play_chess_cnn)
