import numpy as np


class FlowEnv:

    def __init__(self, board, endpoints):
        self.initial_board = [row[:] for row in board]
        self.endpoints = endpoints
        self.reset()

    def reset(self):
        self.board = [row[:] for row in self.initial_board]
        self.done = False

        self.owner = [
            [None for _ in row] for row in self.board
        ]

        for color, points in self.endpoints.items():
            for r, c in points:
                self.owner[r][c] = color
                self.board[r][c] = color

        return self.get_observation()

    def decode_action(self, action):
        num_dirs = 4
        rows = len(self.board)
        cols = len(self.board[0])
        num_cells = rows * cols

        color_list = list(self.endpoints.keys())

        color_idx = action // (num_cells * num_dirs)
        rem = action % (num_cells * num_dirs)

        cell_idx = rem // num_dirs
        dir_idx = rem % num_dirs

        color = color_list[color_idx]

        r = cell_idx // cols
        c = cell_idx % cols

        directions = ["up", "down", "left", "right"]
        direction = directions[dir_idx]

        return color, r, c, direction

    def step(self, action):
        color, row, col, direction = self.decode_action(action)

        if self.done:
            return self.get_observation(), 0, True, {}

        if self.owner[row][col] is None:
            return self.get_observation(), -10, False, {}

        if self.owner[row][col] != color:
            return self.get_observation(), -10, False, {}

        if direction == "up":
            nr, nc = row - 1, col
        elif direction == "down":
            nr, nc = row + 1, col
        elif direction == "left":
            nr, nc = row, col - 1
        elif direction == "right":
            nr, nc = row, col + 1
        else:
            return self.get_observation(), -1, False, {}

        if nr < 0 or nr >= len(self.board) or nc < 0 or nc >= len(self.board[0]):
            return self.get_observation(), -5, False, {}

        if self.owner[nr][nc] is not None:
            return self.get_observation(), -5, False, {}

        if self.board[nr][nc] not in [None, color]:
            return self.get_observation(), -10, False, {}

        self.board[nr][nc] = color
        self.owner[nr][nc] = color

        reward = 1

        if self.check_done():
            self.done = True
            reward += 100

        return self.get_observation(), reward, self.done, {}

    def check_done(self):
        for row in self.board:
            for cell in row:
                if cell is None:
                    return False
        return True

    def print_board(self):
        for row in self.board:
            print(" ".join(cell if cell is not None else "." for cell in row))

    def get_observation(self):
        rows = len(self.board)
        cols = len(self.board[0])

        obs = np.zeros((5, rows, cols), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                cell = self.board[r][c]

                if cell is None:
                    obs[0, r, c] = 1
                elif cell == "R":
                    obs[1, r, c] = 1
                elif cell == "B":
                    obs[2, r, c] = 1

                if (r, c) in self.endpoints.get("R", []) or (r, c) in self.endpoints.get("B", []):
                    obs[3, r, c] = 1

                if self.owner[r][c] is not None:
                    obs[4, r, c] = 1

        return obs