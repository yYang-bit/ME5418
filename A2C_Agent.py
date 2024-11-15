from A2C_model import ActorCritic
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class A2CAgent:
    def __init__(self, epsilon, lr=0.001, gamma = 0.99, epsilon_decay = 0.999):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def act(self, states):
        states = torch.FloatTensor(states).to(self.device)
        probs, _ = self.model(states)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0,4)
        
        else:
            dist = Categorical(probs)
            
            action = dist.sample().item()
        return action 
        


    def train (self, states, actions, rewards, next_states, dones):
        #states = np.array(states)
        #next_states = np.array(next_states)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        #print("Next State Tensor Shape:", next_states.shape) 
        dones = torch.LongTensor(dones).to(self.device)

        probs, states_value = self.model(states)
        _, next_states_value = self.model(next_states)

        advantages = rewards + self.gamma * next_states_value * (1 - dones) - states_value
        #print(f"action:{actions}")
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)