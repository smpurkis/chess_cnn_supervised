import collections
from copy import deepcopy
from pathlib import Path

import chess
import chess.svg
import numpy as np
from cairosvg import svg2png
from chess import Board
from kivy.app import App
from kivy.clock import Clock
from kivy.cache import Cache

from kivy.graphics.vertex_instructions import Rectangle
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from tensorflow.python.keras.models import load_model

from data_utils import target_to_move
from chess_model import ChessModel


class ChessButton(Button):
    def __init__(self, id, **kwargs):
        super().__init__(**kwargs)
        self.id = id


class BoardButton(Button):
    def __init__(self, board_position, **kwargs):
        self.board_position = board_position
        super().__init__(**kwargs)


class ChessGUI(App):
    def __init__(self, model_path, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.board_size = 8
        self.button_positions = {}
        self.board = Board()

        moves_folder = Path("moves/")
        if not moves_folder.exists():
            moves_folder.mkdir(exist_ok=False)
        svg2png(bytestring=chess.svg.board(self.board, size=350), write_to=f'moves/chess.png')
        self.move_str = ""

        self.engine = ChessModel(model_path, verbose=verbose)

    def build(self):
        self.layout = GridLayout(cols=self.board_size, padding="31px")

        with self.layout.canvas:
            self.rect = Rectangle(source='chess.png')

        Clock.schedule_interval(self.update, 1)
        self.add_board_buttons()
        return self.layout

    def update(self, *args):
        self.rect.size = self.root.size
        self.rect.pos = self.root.pos
        self.rect.size = (self.rect.size[0] + 0.1, self.rect.size[1])

    def reset_board(self):
        [Path(m).unlink(missing_ok=True) for m in list(Path("moves/").glob("chess*"))]
        self.board = Board()
        self.move_str = ""
        save_name = 'chess.png'
        svg2png(bytestring=chess.svg.board(self.board, size=350),
                write_to=save_name)
        self.rect.source = save_name

    def check_pawn_promotion(self, move):
        move_str = str(move)
        move_from = move_str[:2]
        piece_to_move = str(self.board.piece_at(chess.parse_square(move_from)))
        if piece_to_move not in ["p", "P"]:
            return move
        move_to = move_str[2:]
        if move_from[1] in ["2", "7"] and move_to[1] in ["1", "8"]:
            move.promotion = chess.QUEEN
        return move

    def update_button(self, button):
        self.move_str += button.id
        # print(self.move_str)

        if len(self.move_str) == 4:
            Cache.remove('kv.image')
            Cache.remove('kv.texture')
            move = chess.Move(
                from_square=chess.parse_square(self.move_str[:2]),
                to_square=chess.parse_square(self.move_str[2:4])
            )
            move = self.check_pawn_promotion(move)
            print(f"Your move: {move}")
            if move in self.board.legal_moves:
                self.board.push(move)
                [Path(m).unlink(missing_ok=True) for m in list(Path("moves/").glob("chess*"))]
                save_name = f'moves/chess_{str(move)}_{self.board.turn}.png'
                svg2png(bytestring=chess.svg.board(self.board, size=350),
                        write_to=save_name)
                self.rect.source = save_name

                if self.board.is_game_over():
                    return self.declare_winner()

                # ai_move = np.random.choice(list(self.board.legal_moves))
                # ai_move = list(self.board.legal_moves)[0]
                # print(ai_move)
                ai_move = self.engine.make_move(self.board)
                print(f"AI move: {ai_move}")

                if isinstance(ai_move, chess.Move):
                    self.board.push(ai_move)
                else:
                    return self.declare_winner()

                if self.board.is_game_over():
                    return self.declare_winner()
                [Path(m).unlink(missing_ok=True) for m in list(Path("moves/").glob("chess*"))]
                save_name = f'moves/chess_{str(move)}_{self.board.turn}.png'
                svg2png(bytestring=chess.svg.board(self.board, size=350),
                        write_to=save_name)
                self.rect.source = save_name
            self.move_str = ""

    def board_to_game(self):
        board = self.board
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
        Path("games").open("a").write(str(game))
        return game

    def declare_winner(self):
        popup = Popup(title='Test popup',
                      content=Label(text=f"{self.board.result()}"),
                      size_hint=(None, None), size=(400, 400))
        popup.open()
        self.board_to_game()
        self.reset_board()

    def add_board_buttons(self):
        for i in reversed(range(8)):
            for j in range(8):
                k = i * 8 + j
                self.layout.add_widget(
                    ChessButton(
                        id=str(target_to_move(k))[:2],
                        background_color=(0.1, 0.1, 0.1, 0.5),
                        on_press=self.update_button
                    )
                )


if __name__ == '__main__':
    model_path = Path("model_name.txt").open("r").read()
    ChessGUI(model_path, verbose=True).run()
