from A2C_model import ActorCritic
import torch
from env import SumoGym
from A2C_Agent import A2CAgent



if __name__=='__main__':
    env = SumoGym()
    model = ActorCritic()
    #model = DQN()
    agent = A2CAgent()
    #agent = DQN()
    step = 0
        #Observation: (ego_position, ego_lane_id, ego_velocity, left_lane_availability, right_lane_availability, can_change_lane)
        #state = torch.randn(batch_size, 6, dtype = torch.float32)
        #actor,critic = model(state)
    state = env.reset()
        #print(f"{state}")
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for i in range(50):
        action = agent.act(state)
        #print(f"{action}")
        next_state,reward,done,_ = env.step(action)   
        print(f"step: {step}, state: {state}, action: {action}, next_state: {next_state}, reward: {reward}, done= {done}")
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

        agent.train(states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state = next_state
        reward += reward
        step += 1
