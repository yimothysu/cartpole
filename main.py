import random
from collections import deque

import torch
from torch import nn
import gym

class QLearner(nn.Module):
    def __init__(self, input_size=4, output_size=2, epsilon=1.0, train_start=1000):
        super().__init__()
        
        self.train_start = train_start
        self.epsilon = epsilon
        self.epsilon_decay = .998
        self.gamma = .95

        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def act(self, observation, iteration):
        if iteration > self.train_start:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < .01:
                self.epsilon = 0
        return random.choice([0, 1]) if random.random() < self.epsilon else torch.argmax(self(torch.Tensor(observation))).item()

    def learn(self, record, loss_fn, optimizer):
        observation, action, observation_next, reward, terminated = record
        if terminated:
            reward = -100

        pred = self(torch.Tensor(observation))
        v_next = torch.max(
            self(torch.Tensor(observation_next))
        )
        y = torch.clone(pred)
        y[action] = reward if terminated else reward + self.gamma * v_next

        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class ReplayBuffer:
    def __init__(self, train_start=1000):
        self.table = deque(maxlen=2000)
        self.train_start = train_start
    
    def append(self, state, action, state_next, reward, terminated):
        self.table.append((state, action, state_next, reward, terminated))
    
    def sample(self, batch_size=64):
        if len(self) < self.train_start:
            return []
        return random.sample(self.table, min(len(self), batch_size))
    
    def __len__(self):
        return len(self.table)

env = gym.make("CartPole-v1", render_mode="human")

learner = QLearner()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(learner.parameters(), lr=1e-3)

replay_buffer = ReplayBuffer()
terminated = False

EPISODES = 1000
for episode in range(EPISODES):
    if episode % 10 == 0:
        print(f"---")

    observation, info = env.reset()
    done = False

    score = 0
    while not done:
        score += 1

        env.render()

        action = learner.act(observation, len(replay_buffer))
        observation_next, reward, terminated, truncated, info = env.step(action)
        replay_buffer.append(observation, action, observation_next, reward, terminated)

        sample = replay_buffer.sample()
        for record in sample:
            learner.learn(record, loss_fn, optimizer)

        observation = observation_next
        done = terminated or truncated

    print(f"episode {episode}: score {score}")

env.close()
