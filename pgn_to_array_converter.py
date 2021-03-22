# import tensorflow as tf
import shutil
import sys
import zipfile
from pathlib import Path

import chess.pgn
import chess.pgn
import numpy as np
import ray
import wget
from tqdm import tqdm

from data_utils import move_to_target, board_to_tensor, tensor_to_board, target_to_move

# print(f"resursion limit {10000 * sys.getrecursionlimit()}")
sys.setrecursionlimit(10000 * sys.getrecursionlimit())


def run_game_normal(game):
    white_win = dict(game.headers)["Result"].split("-")[0] == "1"
    black_win = dict(game.headers)["Result"].split("-")[0] == "0"
    x_board = []
    x_meta = []
    y = []
    if not white_win and not black_win:
        return x_board, x_meta, y
    board = game.board()
    tensor, meta = board_to_tensor(board)
    board_check = chess.Board(tensor_to_board(tensor, meta))
    assert str(board) == str(board_check)
    assert board.turn == board_check.turn
    for i, move in enumerate(game.mainline_moves()):
        target = move_to_target(move)
        if white_win and board.turn or black_win and not board.turn:
            x_board.append(tensor)
            x_meta.append(meta)
            y.append(target)
        move_check = target_to_move(target)
        assert str(move)[:4] == str(move_check)
        board.push(move)
        tensor, meta = board_to_tensor(board)
        board_check = chess.Board(tensor_to_board(tensor, meta))
        assert str(board) == str(board_check)
        assert board.turn == board_check.turn
    return x_board, x_meta, y


def download_pgn(download_url, download=True, overwrite=True):
    output_zip = Path("data/pgn_games.zip")
    if output_zip.exists() and not overwrite:
        keep_zip = input(f"{output_zip.name} detected. Would you like to use this? (y/n) ")
        if keep_zip == "n":
            output_zip.unlink(missing_ok=True)
            download = True
        else:
            download = False
    if download:
        print(f"Downloading PGN at: {download_url}")
        wget.download(download_url, out=output_zip.as_posix())
    zip_file = zipfile.ZipFile(output_zip.as_posix())
    zip_file.extractall("data")
    pgn_path = Path("data", zip_file.filelist[0].filename)
    assert pgn_path.exists()
    return pgn_path


class PgnToArrayConverter:
    def __init__(self, pgn_file, rating=2600, game_limit=1000000000, batch_number=10000):
        self.rating = rating
        self.game_limit = game_limit
        self.batch_number = batch_number
        self.pgn_file = pgn_file
        self.game_indices = []

    def pgn_to_arrays(self):
        print("Indexing games")
        self.get_game_indices()
        print("Converting to arrays")
        self.convert_games_to_arrays()
        print("Merging data")
        self.make_data_to_one_file()

    def get_game_indices(self):
        game_number = 0
        pgn_indices = self.pgn_file.open("r")
        games_indices = []
        header = chess.pgn.read_headers(pgn_indices)
        while header:
            if game_number > self.game_limit:
                break
            # if game_number % 100000 == 0:
            #     print(game_number)
            black_elo = dict(header)["BlackElo"]
            white_elo = dict(header)["WhiteElo"]
            black_elo = int(black_elo) if black_elo.isalnum() else 0
            white_elo = int(white_elo) if white_elo.isalnum() else 0
            termination = dict(header)["Termination"]
            if black_elo > self.rating or white_elo > self.rating and termination != "Time forfeit":
                games_indices.append(game_number)
            header = chess.pgn.read_headers(pgn_indices)
            game_number += 1
        print(f"Number of training games: {len(games_indices)}")
        self.game_indices = games_indices

    @staticmethod
    @ray.remote
    def run_game(game):
        return run_game_normal(game)

    def convert_games_to_arrays(self):
        [arr_file.unlink(missing_ok=True) for arr_file in Path("data", "splits").glob("*")]
        ray.init()
        pgn = self.pgn_file.open("r")
        number_of_chucks = int(((self.game_indices[-1] + 1) / self.batch_number)) + 1
        chucks = [range(self.batch_number * j, min(self.batch_number * (j + 1), self.game_limit)) for j in
                  range(0, number_of_chucks - 1)]
        pbar = tqdm(chucks)
        for k, chuck in enumerate(pbar):
            games = []
            # print(k, len(chucks), chuck)
            for i in chuck:
                if i in self.game_indices:
                    game = chess.pgn.read_game(pgn)
                    games.append(game)
                else:
                    chess.pgn.skip_game(pgn)
            # s = time()
            tasks = [self.run_game.remote(game) for game in games]
            results = ray.get(tasks)
            # print(f"{time() - s}")
            # s = time()
            # results = [run_game_normal(game) for game in games]
            # print(f"{time() - s}")
            # s = time()
            tensors = [r[0] for r in results]
            tensors = [item for move_series in tensors for item in move_series]
            tensors = np.asarray(tensors)
            metas = [r[1] for r in results]
            metas = [item for move_series in metas for item in move_series]
            metas = np.asarray(metas)
            targets = [r[2] for r in results]
            targets = [item for move_series in targets for item in move_series]
            targets = np.asarray(targets)
            # print(f"{time() - s}")

            tensors_file = Path(f"data/splits/tensors_{self.rating}_or_greater_{k}")
            tensors_file.parent.mkdir(parents=True, exist_ok=True)
            metas_file = Path(f"data/splits/metas_{self.rating}_or_greater_{k}")
            metas_file.parent.mkdir(parents=True, exist_ok=True)
            targets_file = Path(f"data/splits/targets_{self.rating}_or_greater_{k}")
            targets_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(tensors_file.as_posix(), tensors)
            np.save(metas_file.as_posix(), metas)
            np.save(targets_file.as_posix(), targets)
        ray.shutdown()

    @staticmethod
    def load_to_one_file(save_name, files):
        total_lengths = []
        array_files = []
        for file in files:
            file_arr = np.load(str(file), allow_pickle=True)
            if file_arr.shape[0] > 0:
                arr_shape = file_arr.shape
                array_files.append(file)
                total_lengths.append(file_arr.shape[0])
        if len(total_lengths) == 0:
            raise Exception("No games selected with these settings.")
        nmemmap_shape = (sum(total_lengths), *arr_shape[1:])
        nmemmap = np.memmap(f"data/{save_name}", shape=nmemmap_shape, mode="w+", dtype=np.int16)
        start = 0
        end = 0
        for i, file in enumerate(array_files):
            end += total_lengths[i]
            file_arr = np.load(str(file), allow_pickle=True)
            nmemmap[start: end] = file_arr
            nmemmap.flush()
            start += total_lengths[i]

    def make_data_to_one_file(self):
        split_files = list(Path("data", "splits").glob("*"))
        tensor_files = sorted([f for f in split_files if "tensor" in f.name])
        meta_files = sorted([f for f in split_files if "meta" in f.name])
        target_files = sorted([f for f in split_files if "target" in f.name])

        print("Processing tensors")
        self.load_to_one_file(f"tensors", tensor_files)
        print("Saving tensors")
        print("Processing metas")
        self.load_to_one_file(f"metas", meta_files)
        print("Saving metas")
        print("Processing targets")
        self.load_to_one_file(f"targets", target_files)
        print("Saving targets")
        shutil.rmtree("./data/splits/", ignore_errors=True)


if __name__ == '__main__':
    pgn_file = download_pgn("https://database.nikonoel.fr/lichess_elite_2021-01.zip")
    data_converter = PgnToArrayConverter(pgn_file)
    data_converter.pgn_to_arrays()
