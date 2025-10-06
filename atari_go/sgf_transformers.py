import pandas as pd
import numpy as np
from sgfmill import sgf
from sgfmill.boards import Board

def board_to_array(b):
    size = b.side
    arr = np.zeros((size, size), dtype=int)
    for row in range(size):
        for col in range(size):
            stone = b.get(row, col)
            if stone == 'b':
                arr[row, col] = -1
            elif stone == 'w':
                arr[row, col] = 1
    return arr

def get_lost_stones(before_board, after_board, color):
    lost = []
    for row in range(before_board.side):
        for col in range(before_board.side):
            if before_board.get(row, col) == color and after_board.get(row, col) != color:
                lost.append((row, col))
    return lost

def sgf_to_dataframe(sgf_path):
    with open(sgf_path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    board_size = game.get_size()
    board = Board(board_size)
    node = game.get_root()

    data = []
    black_captures = 0
    white_captures = 0

    prev_board = board.copy()

    for node in game.get_main_sequence()[1:]:  # skip root
        move = node.get_move()
        color, row_column = move
        
        (row, col) = row_column if row_column is not None else (None, None)

        if color is None or row is None:
            continue
        
        before_board = board.copy()

        try:
            board.play(row, col, color)
        except ValueError:
            continue  # illegal move, found this in the sgfmaill library 

        after_board = board.copy()

        lost_stones = get_lost_stones(prev_board, before_board, color)

        captured_now = get_lost_stones(before_board, after_board, 'w' if color == 'b' else 'b')

        if color == 'b':
            black_captures += len(captured_now)
        else:
            white_captures += len(captured_now)

        data.append({
            'player': color,
            'move': (row, col),
            'before': before_board,
            'after': after_board,
            'lost_stones_before': lost_stones,
            'black_captures': black_captures,
            'white_captures': white_captures,
        })

        prev_board = before_board

    df = pd.DataFrame(data)
    return df

def sgf_to_csv(sgf_path, csv_path):
    df = sgf_to_dataframe(sgf_path)
    df.to_csv(csv_path, index=False)
