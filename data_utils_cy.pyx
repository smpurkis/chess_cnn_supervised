# cython: infer_types=True
cimport cython
import chess.pgn
import numpy as np
cimport numpy as np
from cpython cimport array
import array
from time import time


cpdef tuple board_to_tensor(board):
    cdef:
        str fen
        str metadata
        list metadata_list
        np.ndarray arr
        np.ndarray meta_arr
        list rows
        int j
        int i
        str p
        int l
        list piece_list
        bint k
        str m
    fen = board.fen()
    metadata = "".join(fen.split()[1:3])
    arr = np.zeros(shape=(8, 8, 8), dtype=np.int16)
    rows = fen.split()[0].split("/")
    j = 0
    for row in rows:
        i = 0
        for p in row:
            if p.isalpha():
                p_lower = p.lower()
                piece_list = [p.isupper(),
                             p_lower == "p",
                             p_lower == "n",
                             p_lower == "b",
                             p_lower == "r",
                             p_lower == "q",
                             p_lower == "k",
                             p.isupper()]
                for l, k in enumerate(piece_list):
                    arr[j][i][l] = k
                i += 1
            elif p.isnumeric():
                i += int(p)
        j += 1
    metadata_list = [1 if m not in ["w", "b"] else 1 if m == "w" else 0 for m in metadata]
    meta_arr = np.asarray(metadata_list)
    meta_arr = np.pad(meta_arr, (0, 8 - len(metadata_list)))
    # arr = np.append(arr, meta_arr).reshape((9, 8))
    return arr, meta_arr


cpdef move_to_target(move):
    move = str(move)[:4]
    t = np.asarray([int(ord(m)) - ord("a") if m.isalpha() else int(m) - 1 for m in move])
    t = np.asarray([t[i] * (8 ** i) for i in range(len(t))]).sum()
    return t


cpdef target_to_move(int target):
    cdef:
        int i
        np.ndarray m
    if target == 0:
        return "a1a1"
    m = np.zeros(4, dtype=np.int)
    for i in range(4):
        m[3-i] = np.floor(target / (8 ** (3-i)))
        target = target % (8 ** (3-i))
    from_square = chess.parse_square(chr(int(m[0]) + ord("a")) + str(m[1]+1))
    to_square = chess.parse_square(chr(m[2] + ord("a")) + str(m[3]+1))
    move = chess.Move(from_square=from_square, to_square=to_square)
    return move


cpdef str tensor_to_board(np.ndarray arr, np.ndarray meta_arr):
    cdef:
        str fen
        int s
        np.ndarray row
        np.ndarray piece_arr
        int isUpper
        int p_index
        str p
        int piece_arr2
        int m
        list castling
    fen = ""
    s = 0
    piece_key = ["p", "n", "b", "r", "q", "k"]
    for row in arr:
        for piece_arr in row:
            if len(piece_arr.nonzero()[0]) == 0:
                s += 1
            else:
                if s > 0:
                    fen += str(s)
                    s = 0
                isUpper = piece_arr[0]
                p_index = np.where(piece_arr[1:7] == 1)[0][0]
                p = piece_key[p_index]
                if isUpper:
                    p = p.upper()
                fen += p
        if s > 0:
            fen += str(s)
            s = 0
        fen += "/"
    fen = fen.rstrip("/")
    castling = ["K", "Q", "k", "q"]
    for piece_arr2, m in enumerate(meta_arr):
        if piece_arr2 == 0:
            if m == 1:
                fen += " w "
            else:
                fen += " b "
        else:
            if m != 0:
                fen += castling[piece_arr2 - 1]
            else:
                continue
    return fen


cpdef tuple run_game_normal(game):
    cdef :
        list x_board = []
        list x_meta = []
        list y = []
        np.ndarray tensor
        np.ndarray meta
    board = game.board()
    tensor, meta = board_to_tensor(board)
    board_check = chess.Board(tensor_to_board(tensor, meta))
    assert str(board) == str(board_check)
    assert board.turn == board_check.turn
    for i, move in enumerate(game.mainline_moves()):
        target = move_to_target(move)
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
