import random

import chess
import inquirer
import ray
import time


@ray.remote
def ray_play_random_game(results):
    return play_random_game(results)


def play_random_game(results):
    board = chess.Board()
    # print(board)
    turn_number = 0
    while not board.is_game_over():
        turn_number += 1
        if board.turn == chess.WHITE and False:
            questions = [
                inquirer.List(
                    "move",
                    message="What is your move?",
                    choices=list(board.legal_moves),
                )
            ]
            move = inquirer.prompt(questions)
            board.push(move["move"])
        else:
            move = random.choice(list(board.legal_moves))
            board.push(move)
        # print(board)
        # print(turn_number)
        # print(board.turn)
    # print("statemate:", board.is_stalemate())
    # print("fivefold repetition:", board.is_fivefold_repetition())
    # print("75 moves:", board.is_seventyfive_moves())
    # print("3th repitition:", board.is_repetition())
    # print("insufficient material:", board.is_insufficient_material())
    if not any(
        (
            board.is_stalemate(),
            board.is_fivefold_repetition(),
            board.is_seventyfive_moves(),
            board.is_repetition(),
            board.is_insufficient_material(),
        )
    ):
        if board.turn == chess.WHITE:
            results["black"] += 1
        else:
            results["white"] += 1
    else:
        results["draw"] += 1
    # print(results)
    return results


if __name__ == "__main__":
    results = {"white": 0, "black": 0, "draw": 0}
    start = time.time()
    ray.init()
    futures = [ray_play_random_game.remote(results) for i in range(10)]
    results_list = ray.get(futures)
    results = {"white": 0, "black": 0, "draw": 0}
    for r in results_list:
        for k, v in r.items():
            results[k] += v
    # results = [play_random_game(results) for i in range(100)][0]
    print(results)
    o = 0
