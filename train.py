import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from flow_env import FlowEnv
from model import PolicyNet
from replay_buffer import ReplayBuffer


gamma = 0.9
batch_size = 32
start_learning = 200


board = [
    ["R", None, None, "B"],
    [None, None, None, None],
    ["R", None, None, "B"]
]

endpoints = {
    "R": [(0, 0), (2, 0)],
    "B": [(0, 3), (2, 3)]
}

env = FlowEnv(board, endpoints)
obs = env.reset()

num_actions = len(endpoints) * len(board) * len(board[0]) * 4


policy_net = PolicyNet(num_actions, obs.shape)
target_net = PolicyNet(num_actions, obs.shape)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

memory = ReplayBuffer()


def get_valid_actions(env, num_actions, endpoints):
    valid = []

    rows = len(env.board)
    cols = len(env.board[0])

    colors = list(endpoints.keys())
    directions = ["up", "down", "left", "right"]

    for color_idx, color in enumerate(colors):
        for r in range(rows):
            for c in range(cols):

                if env.owner[r][c] != color:
                    continue

                for d_idx, direction in enumerate(directions):
                    nr, nc = r, c

                    if direction == "up":
                        nr -= 1
                    elif direction == "down":
                        nr += 1
                    elif direction == "left":
                        nc -= 1
                    elif direction == "right":
                        nc += 1

                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue

                    if env.owner[nr][nc] is not None:
                        continue

                    action = (
                        color_idx * rows * cols * 4
                        + r * cols * 4
                        + c * 4
                        + d_idx
                    )

                    valid.append(action)

    return valid


episodes = 500

for episode in range(episodes):

    obs = env.reset()
    total_reward = 0
    done = False

    while not done:

        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)

        q_values = policy_net(obs_tensor)

        valid_actions = get_valid_actions(env, num_actions, endpoints)

        if len(valid_actions) == 0:
            action = random.randint(0, num_actions - 1)
        else:
            if random.random() < 0.1:
                action = random.choice(valid_actions)
            else:
                action = valid_actions[
                    torch.argmax(q_values[0, valid_actions]).item()
                ]

        next_obs, reward, done, _ = env.step(action)

        memory.push(obs, action, reward, next_obs, done)

        obs = next_obs
        total_reward += reward

        if len(memory) < start_learning:
            continue

        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        q_values = policy_net(states)

        with torch.no_grad():
            next_q_values = target_net(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]

        target_q = q_values.clone().detach()

        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += gamma * max_next_q[i]
            target_q[i][actions[i]] = target

        loss = loss_fn(q_values, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print("Episode:", episode, "Reward:", total_reward)