from A2C_model import ActorCritic
import torch
from env import SumoGym
from A2C_Agent import A2CAgent
import os
import matplotlib.pyplot as plt


if __name__=='__main__':
    env = SumoGym(show_gui=True)
    model = ActorCritic()
    #model = DQN()
    agent = A2CAgent(epsilon=0)
    #agent = DQN()
        #Observation: (ego_position, ego_lane_id, ego_velocity, left_lane_availability, right_lane_availability, can_change_lane)
        #state = torch.randn(batch_size, 6, dtype = torch.float32)
        #actor,critic = model(state)
    max_episodes = 3000
    max_steps = 100
    train_interval = 5
    episode_reward = []
    states, actions, rewards, next_states, dones = [], [], [], [], []

    model_path = './models/new_A2C_3000.pth' 
    agent.model.load_state_dict(torch.load(model_path, weights_only=True))
    #agent.model.train()  
    agent.model.eval() 

    for episode in range(max_episodes+1):
        episode += 1
        print(f"start epsidode: {episode}")
        total_reward = 0
        state = env.reset()
        next_state,_,_ = env.step(2)
        state = next_state

           
        #states, actions, rewards, next_states, dones = [], [], [], [], []

        for step in range(max_steps):
            action = agent.act(state)
                #print(f"{action}")
            next_state,reward,done = env.step(action)
                # if 'car_0' not in traci.vehicle.getIDList():
                #     print("Vehicle 'car_0' is not in the simulation. Ending the episode.")
                # break  # 结束当前 episode
                # state_str = ', '.join(f"{s:.2f}" for s in state)  # 格式化 state
                # next_state_str = ', '.join(f"{ns:.2f}" for ns in next_state)  # 格式 
                # print(f"step: {step}, state: {state_str}, action: {action}, next_state: {next_state_str}, reward: {reward:.2f}, done= {done}")
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            step += 1
            total_reward += reward

            if len(states) >= train_interval:
                print(f"step: {step},\nstate: {states},\naction: {actions},\nnext_state: {next_states},\nreward: {rewards},\ndone= {dones}\n")
                agent.train(states, actions, rewards, next_states, dones)
                states, actions, rewards, next_states, dones = [], [], [], [], []
                
            if done: 
                if len(states) > 0:
                    print(f"step: {step},\nstate: {states},\naction: {actions},\nnext_state: {next_states},\nreward: {rewards},\ndone= {dones}\n")
                    agent.train(states, actions, rewards, next_states, dones)
                    states, actions, rewards, next_states, dones = [], [], [], [], []    
                break
                    
        episode_reward.append(total_reward)
        env.close()
                
        # if episode % 500 == 0:
        #     save_dir = './models'
        #     save_path = os.path.join(save_dir, f"new_A2C_{episode}.pth")
        #     torch.save(agent.model.state_dict(), save_path)

        # with open('episode_reward_8.txt','a') as f:
        #     f.write(f"total steps are {step}, reward is {total_reward} in epsidode {episode} \n")


