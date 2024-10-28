import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size=6, action_size=5):
        super(DQN, self).__init__()
        
        # Q Network Structure
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 64),  # Input Layer -> 64 Features
            nn.ReLU(),                  # Activation Function
            nn.Linear(64, 64),          # Hidden Layer -> 64 Features
            nn.ReLU(),
            nn.Linear(64, action_size)  # Output Layer -> Action Space Size（5）
        )

    def forward(self, x):
        return self.q_network(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001, memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        # Q Network and Optimizer
        self.q_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q Value and Target Q Value
        current_q_values = self.q_network(states).gather(1, actions).squeeze()
        next_q_values = self.q_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Loss Function
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        # Update Q Network Parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Dynamically Adjust Epsilon Value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
