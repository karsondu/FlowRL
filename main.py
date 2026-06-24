from flow_env import FlowEnv

board = [
    ["R", ".", ".", "B"],
    [".", ".", ".", "."],
    ["R", ".", ".", "B"]
]

endpoints = {
    "R": [(0,0), (2,0)],
    "B": [(0,3), (2,3)]
}
env = FlowEnv(board,endpoints)

env.print_board()
print(env.step(("R",0,0,"right")))
print(env.step(("R",0,1,"right")))

