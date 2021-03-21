import random
import time

import chess
import ray
from tensorflow.python.keras.models import load_model

from chess_model import ChessModel


@ray.remote
def play_game_remote(ai, results, i):
    ai = load_model("models/epochs_40_batch_size_512_2021-03-19_19:44:22.336811.h5")
    return play_game(ai, results, i)


def play_game(ai, results, i):
    print(i)
    board = chess.Board()
    # print(board)
    turn_number = 0
    while not board.is_game_over():
        turn_number += 1
        # print(f"turn number: {turn_number}")
        if board.turn == chess.WHITE:
            move = random.choice(list(board.legal_moves))
            board.push(move)
            # questions = [
            #     inquirer.List("move",
            #                   message="What is your move?",
            #                   choices=list(board.legal_moves),
            #                   )
            # ]
            # move = inquirer.prompt(questions)
            # board.push(move["move"])
        else:
            # ai_moves = get_legal_ai_moves(ai, board)
            # if len(ai_moves) > 0:
            #     ai_move = ai_moves[0]
            #     board.push(ai_move)
            # else:
            #     move = random.choice(list(board.legal_moves))
            #     board.push(move)
            ai_move = ChessModel(ai).make_move(board)
            board.push(ai_move)
            # move = random.choice(list(board.legal_moves))
            # board.push(move)

    if not any((board.is_stalemate(),
                board.is_fivefold_repetition(),
                board.is_seventyfive_moves(),
                board.is_repetition(),
                board.is_insufficient_material())):
        if board.turn == chess.WHITE:
            results["black"] += 1
        else:
            results["white"] += 1
    else:
        results["draw"] += 1
    # print(results)
    return results


if __name__ == '__main__':
    # ai = load_model("models/epochs_40_batch_size_512_2021-03-19_19:44:22.336811.h5")
    ai = "models/epochs_40_batch_size_512_2021-03-19_19:44:22.336811.h5"
    results = {
        "white": 0,
        "black": 0,
        "draw": 0
    }
    ray.init()
    start = time.time()
    futures = [play_game_remote.remote(ai, results, i) for i in range(100)]
    results_list = ray.get(futures)
    print(f"{time.time() - start}")
    results = {
        "white": 0,
        "black": 0,
        "draw": 0
    }
    for r in results_list:
        for k, v in r.items():
            results[k] += v
    ai = load_model("models/epochs_40_batch_size_512_2021-03-19_19:44:22.336811.h5")
    start = time.time()
    # results2 = [play_game(ai, results, i) for i in range(100)][0]
    print(f"{time.time() - start}")
    print(results)
    # print(results2)
    o = 0
