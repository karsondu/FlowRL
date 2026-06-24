class FlowEnv():

    def __init__(self,board, endpoints):
        self.initial_board = board
        self.endpoints = endpoints
        self.reset()

    def reset(self):
        self.board = [row[:] for row in self.initial_board]
        self.done = False
        self.owner = [
            ["." for _ in row] for row in self.board
        ]

        for color, points in self.endpoints.items():
            for r,c in points:
                self.owner[r][c] = color
        return self.board
    
    def step(self,action):
        color, row, col, direction = action

        if self.done:
            return self.board, 0 , True, {"msg": "already done"}
        if self.owner[row][col] not in [".",color]:
            return self.board, -10, False, {"msg": "not your path"}

        #next direction computations
        if direction == "up":
            nr, nc = row - 1, col
        elif direction == "down":
            nr, nc = row + 1, col
        elif direction == "left":
            nr, nc = row, col - 1
        elif direction == "right":
            nr, nc = row, col + 1
        else: 
            return self.board, -1 , False, {"msg":"invalid direction"}
    
        #bounds checking
        if nr < 0 or nr >= len(self.board) or nc < 0 or nc >= len(self.board[0]):
            return self.board, -5, False, {"msg": "out of bounds"}
        
        #collision checking 
        if self.board[nr][nc] != ".":
            return self.board, -5, False, {"msg": "blocked cell"}
        
        #prevent touching wrong color
        if self.board[nr][nc] not in [".", color]:
            return self.board, -10, False, {"msg": "wrong color cell"}
        
        self.board[nr][nc] = color
        self.owner[nr][nc] = color

        reward = 1

        if self.check_done():
            self.done = True
            reward += 100

        return self.board, reward, self.done, {}
    
    def check_done(self):
        for row in self.board:
            if "." in row:
                return False
        return True





    def print_board(self):
        for row in self.board:
            print(" ".join(row))