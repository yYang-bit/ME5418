import torch
import numpy as np
from A2C_model import ActorCritic
from env import SumoGym
from torch.distributions import Categorical
import matplotlib.pyplot as plt

def test_episodes():
    model_path = './models/new_A2C_3000.pth' 
    model = ActorCritic()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 初始化环境
    env = SumoGym(show_gui=False)  

    all_episode_speeds = []
    all_episode_steps = []
    
    for episode in range(10):
        state = env.reset()
        done = False
        speeds = []
        step_count = 0
        
        while not done and step_count < 101:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                probs, _ = model(state_tensor)
                dist = Categorical(probs)
                action = dist.sample().item()
            
            next_state, reward, done = env.step(action)
            
            speed = next_state[1] * 4.5
            speeds.append(speed)
            
            state = next_state
            step_count += 1
        
        all_episode_speeds.append(speeds)
        all_episode_steps.append(step_count)
        
        print(f"Episode {episode + 1} completed with {step_count} steps")
        env.close()
    

    with open('speed_data.txt', 'w') as f:
        for episode, speeds in enumerate(all_episode_speeds):
            f.write(f"Episode {episode + 1}:\n")
            for step, speed in enumerate(speeds):
                f.write(f"Step {step + 1}: {speed}\n")
            f.write("\n")
    

    plt.figure(figsize=(12, 6))
    for episode, speeds in enumerate(all_episode_speeds):
        steps = range(len(speeds))
        plt.plot(steps, speeds, label=f'Episode {episode + 1}')
    
    plt.title('Vehicle Speed Over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Speed')
    plt.grid(True)
    plt.legend()
    plt.savefig('speed_curves.png')
    plt.show()
    
    env.close()

if __name__ == "__main__":
    test_episodes()