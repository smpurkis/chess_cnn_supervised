from datetime import datetime

import fire

from chess_dataset import ChessDataset
from kivy_gui import ChessGUI
from pgn_to_array_converter import download_pgn, PgnToArrayConverter
from train_chess_cnn_keras import define_model, define_callbacks


def get_data(download_url):
    # download the pgn
    pgn_file = download_pgn(download_url)
    data_converter = PgnToArrayConverter(pgn_file)
    data_converter.pgn_to_arrays()


def train_cnn(model_save_name=None, batch_size=512, epochs=20):
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
    if model_save_name is None:
        model_save_name = f"models/epochs_{epochs}_batch_size_{batch_size}_{str(datetime.now()).replace(' ', '_')}.h5"
    model.save()


def play_cnn(model_path):
    chess_gui = ChessGUI(model_path)
    chess_gui.run()


def download_train_play_chess_cnn(download_url: str = "https://database.nikonoel.fr/lichess_elite_2021-01.zip"):
    """
    The complete pipeline of downloading PGN Zip from Lichess (https://database.nikonoel.fr),
    Training a CNN using Keras (https://keras.io) and then Playing the Engine using Kivy (https://kivy.org).
    :param download_url: Path to Lichess PGN Zip, E.g. https://database.nikonoel.fr/lichess_elite_2021-01.zip".
    Otherwise can modify the get_data function yourself
    """
    model_path = "chess_model.h5"
    get_data(download_url)
    train_cnn(model_path)
    play_cnn(model_path)


if __name__ == '__main__':
    fire.Fire(download_train_play_chess_cnn)
