import collections
import random
from copy import deepcopy
from time import time

import chess
import chess.pgn
import chess.svg
import inquirer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from cairosvg import svg2png

from data_utils import board_to_tensor, target_to_move


def board_to_game(board):
    game = chess.pgn.Game()

    # Undo all moves.
    switchyard = collections.deque()
    while board.move_stack:
        switchyard.append(board.pop())

    game.setup(board)
    node = game

    # Replay all moves.
    while switchyard:
        move = switchyard.pop()
        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = board.result()
    return game


def print_board(board):
    svg2png(bytestring=chess.svg.board(board, size=350), write_to='chess.png')
    p = Image.open("chess.png")
    plt.imshow(np.asarray(p))
    plt.show()


class ChessModel:
    def __init__(self, model_path, verbose=False):
        self.pieces_score = {
            "q": 9,
            "r": 5,
            "b": 3.5,
            "n": 3,
            "p": 1,
            "k": 0
        }
        self.engine = tf.keras.models.load_model(model_path)
        self.verbose = verbose

    def get_legal_ai_moves(self, board, top_n=5):
        # get legal moves
        legal_moves = list(board.legal_moves)
        tensor = list(board_to_tensor(board))
        tensor[0] = tf.reshape(tensor[0], shape=(-1, *tensor[0].shape))
        tensor[1] = tf.reshape(tensor[1], shape=(-1, *tensor[1].shape))

        # get move probabilities from model
        target_probabilities = self.engine.predict(tensor).reshape(-1)

        # rank by probability
        target_rankings = target_probabilities.argsort()[-len(target_probabilities):][::-1]
        move_rankings = np.array([target_to_move(elem) for elem in target_rankings])

        # extracted highest probability legal moves
        legal_move_probabilities = np.array([move for move in move_rankings if move in legal_moves])

        # return top n moves
        return legal_move_probabilities[:top_n]

    def get_move_by_material(self, board, number_of_top_moves=100, depth=1, depth_limit=3, alpha=-100000.0,
                             beta=100000.0, verbose=False):
        # if only want model moves
        if depth_limit == 1:
            return self.get_legal_ai_moves(board, 1)[0]

        piece_map = board.piece_map()
        white_material = np.sum([self.pieces_score[str(p).lower()] for p in piece_map.values() if str(p).isupper()])
        black_material = np.sum([self.pieces_score[str(p).lower()] for p in piece_map.values() if str(p).islower()])

        # calculate score of board state based on material
        current_material_score = white_material - black_material

        # checkmate counts as 10,000
        if board.is_game_over():
            if board.result().split("-")[0] == "1":
                current_material_score = 10000
            elif board.result().split("-")[0] == "0":
                current_material_score = -10000
            return current_material_score
        if depth >= depth_limit:
            return current_material_score
        relative_scores = np.asarray([])

        # get most prioritized_moves
        if depth <= 1:
            top_n = 3
            prioritized_moves = self.get_legal_ai_moves(board, top_n)
            prioritized_moves = np.stack((prioritized_moves[:number_of_top_moves],
                                          np.asarray([depth_limit for i in range(len(prioritized_moves))]),
                                          np.ones(len(prioritized_moves))), axis=1)
            if len(prioritized_moves) != 0:
                all_other_moves = np.asarray(
                    [(move, depth_limit, 0) for move in list(board.legal_moves) if move not in prioritized_moves[:, 0]])
                if all_other_moves.size > 0:
                    prioritized_moves = np.vstack((prioritized_moves, all_other_moves))
            else:
                prioritized_moves = np.asarray([(move, depth_limit, 0) for move in list(board.legal_moves)])
        else:
            prioritized_moves = np.asarray([(move, depth_limit, 0) for move in list(board.legal_moves)])

        # min max search on move tree
        for move, low_depth_limit, priority in prioritized_moves:
            future_board = deepcopy(board)
            future_board.push(move)
            mat_score = self.get_move_by_material(future_board,
                                                  depth=depth + 1,
                                                  depth_limit=low_depth_limit,
                                                  alpha=alpha,
                                                  beta=beta)
            relative_scores = np.append(relative_scores, mat_score)

            # alpha beta pruning
            if board.turn:
                alpha = np.max((alpha, mat_score))
                if beta < alpha:
                    break
            else:
                beta = np.min((beta, mat_score))
                if beta < alpha:
                    break

        if depth > 1:
            # return best score for player
            if board.turn:
                return np.max(relative_scores)
            else:
                return np.min(relative_scores)
        else:
            if self.verbose:
                print(relative_scores)
                print(np.concatenate((prioritized_moves, relative_scores.reshape(-1, 1)), axis=1))

            # return best move result
            if board.turn:
                max_indices = np.where(relative_scores == relative_scores.max())[0]
                if self.verbose:
                    print(max_indices)

                max_priority_indices = np.where(prioritized_moves[max_indices][:, 2] == 1)[0]
                if self.verbose:
                    print(max_priority_indices)

                if max_priority_indices.size > 0:
                    chosen_move_index = max_indices[random.choice(max_priority_indices)]
                    if self.verbose:
                        print(f"max_priority_indices.size {max_priority_indices.size}")
                else:
                    chosen_move_index = random.choice(max_indices)
                if self.verbose:
                    print(f"returning max {prioritized_moves[chosen_move_index]}")
                return prioritized_moves[chosen_move_index][0]
            else:
                min_indices = np.where(relative_scores == relative_scores.min())[0]
                if self.verbose:
                    print(min_indices)

                min_priority_indices = np.where(prioritized_moves[min_indices][:, 2] == 1)[0]
                if self.verbose:
                    print(min_priority_indices)

                if min_priority_indices.size > 0:
                    chosen_move_index = min_indices[random.choice(min_priority_indices)]
                    if self.verbose:
                        print(f"min_priority_indices.size {min_priority_indices.size}")
                else:
                    chosen_move_index = random.choice(min_indices)
                if self.verbose:
                    print(f"returning min {prioritized_moves[chosen_move_index]}")
                return prioritized_moves[chosen_move_index][0]

    def make_move(self, board):
        # move = np.random.choice(legal_move_probabilities[:3])
        s = time()
        move = self.get_move_by_material(
            deepcopy(board),
            depth_limit=3,
            # no_depth=True
        )
        if self.verbose:
            print(f"Get whole move time: {time() - s}")
        # move = legal_move_probabilities[0]
        return move


def play_game():
    ai_model = "models/epochs_40_batch_size_512_2021-03-19_19:44:22.336811.h5"
    engine = ChessModel(ai_model)
    board = chess.Board()
    turn_number = 0
    while not board.is_game_over():
        turn_number += 1
        if board.turn == chess.WHITE:
            print_board(board)
            questions = [
                inquirer.List("move",
                              message="What is your move?",
                              choices=list(board.legal_moves),
                              )
            ]
            move = inquirer.prompt(questions)
            board.push(move["move"])
        else:
            engine.make_move(board)
        print("")
    print(board.result())


if __name__ == '__main__':
    play_game()
