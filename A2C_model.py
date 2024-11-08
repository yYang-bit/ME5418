import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_size=5, action_size=5):
        super(ActorCritic, self).__init__()
        
        # Shared layer for feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_size, 64),  # Input layer with outputting 64 features
            nn.ReLU()                   # Activation function 
        )
        
        # Actor network to output action probability distribution
        self.actor = nn.Sequential(
            nn.Linear(64, action_size),  # Inputting 64 features
            nn.Softmax(dim=-1)           # Activation function 
        )
        
        # Critic network to output the value estimate of the state
        self.critic = nn.Sequential(
            nn.Linear(64, 1)           
        )

    def forward(self, x):
        # Forward pass method
        shared = self.shared(x)          # Extract features 
        actor = self.actor(shared)       # Calculate action probabilities
        critic = self.critic(shared)     # Calculate state value 
        return actor, critic            
