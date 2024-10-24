from model import ActorCritic
import torch



if __name__=='__main__':
    model = ActorCritic()
    batch_size = 10
    #Observation: (ego_position, ego_lane_id, ego_velocity, left_lane_availability, right_lane_availability, can_change_lane)
    state = torch.randn(batch_size, 7, dtype = torch.float32)
    actor,critic = model(state)

    print(f"Action probabilities: \n{actor}")
    print(f"Critic: \n{critic}")