import os
import sys

# check if SUMO_HOME exists in environment variable
# if not, then need to declare the variable before proceeding
# makes it OS-agnostic
if "SUMO_HOME" in os.environ:
   tools = os.path.join(os.environ["SUMO_HOME"], "tools")
   sys.path.append(tools)
else:
   sys.exit("please declare environment variable 'SUMO_HOME'")

#sys.path.append('/home/nusme/.local/lib/python3.8/site-packages')

import gym
from gym import spaces
import numpy as np
import random
from math import inf
import traci
from traci.constants import CMD_GET_VEHICLE_VARIABLE, VAR_POSITION, VAR_LANE_INDEX
from sumolib import checkBinary
import time
import argparse

# Observation: (ego_position, ego_lane_id, ego_velocity, left_lane_availability, right_lane_availability, can_change_lane)
# Availability: 0 - No, 1 - Yes

class SumoGym(gym.Env):
    def __init__(self, show_gui):
        """
        Initialize the SumoGym environment.
        """
        super(SumoGym, self).__init__()

        # Ego vehicle identifier
        self.egoID = "car_0"

        # Simulation parameters
        self.lane_change_duration = 1.5
        self.maximum_speed = 4.5
        self.minimum_speed = 0.5
        self.acceleration = 1
        self.deceleration = 1
        self.step_count = 0
        self.done = False
        self.radius = 6 #10

        # Define observation space: (position, lane_id, velocity, left_availability, right_availability, can_change_lane)
        # low = np.array([-inf, 0, 0, 0, 0, 0], dtype=np.float32)
        # high = np.array([inf, 3, inf, 1, 1, 1], dtype=np.float32)
        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([3, inf, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Define action space: 0 - Change left, 1 - Change right, 2 - Idle, 3 - Accelerate, 4 - Decelerate
        self.action_space = spaces.Discrete(5)

        # SUMO configuration
        self.show_gui = show_gui
        self.sumo_config = "/home/nusme/ME5418/ME5418/sumo_env/ottoman.sumocfg"

    def subscribe_context(self, vehicleID, radius):
        """
        Subscribe to the context of the ego car within the radius.
        
        """
        varList = [
            VAR_POSITION,       # Vehicle position (x, y)
            VAR_LANE_INDEX,     # Lane index
        ]

        traci.vehicle.subscribeContext(
            vehicleID,
            CMD_GET_VEHICLE_VARIABLE,
            radius,
            varList
        )

    def get_subscribed_data(self):
        """
        Retrieve the subscribed context data for the ego vehicle.
        
        Returns:
            dict: A dictionary containing information about nearby vehicles.
        """
        self.subscribe_context(self.egoID, self.radius)
        data = traci.vehicle.getContextSubscriptionResults(self.egoID)

        # Remove ego vehicle from the data if present
        if data and self.egoID in data:
            del data[self.egoID]

        return data

    def get_ego_state(self):
        """
        Get the current state of the ego vehicle.
        
        Returns:
            tuple: A tuple containing position, lane_id, and speed of the ego vehicle.
        """
        speed = traci.vehicle.getSpeed(self.egoID)
        position = traci.vehicle.getPosition(self.egoID)[0]
        lane_id = traci.vehicle.getLaneIndex(self.egoID)
        ego_state = (position, lane_id, speed)
        return ego_state

    def get_observation(self):
        """
        Generate the current observation for the ego car.
        
        Returns:
            tuple: The observation containing ego vehicle state and lane availability.
        """
        # Initialize lane availability: 1 - Available, 0 - Not available
        left_availability = 1
        right_availability = 1
        can_change_lane = 0

        # Get ego vehicle state
        ego_state = self.get_ego_state()
        ego_lane_id = ego_state[1]

        # Check lane boundaries and update availability
        if ego_lane_id == 0:
            right_availability = 0  # Cannot change right if on the leftmost lane
        if ego_lane_id == 3:
            left_availability = 0   # Cannot change left if on the rightmost lane

        # Get data of nearby vehicles
        data = self.get_subscribed_data()

        if data:
            for near_vehicleID, near_vehicle_info in data.items():
                pos = near_vehicle_info.get(VAR_POSITION, (None, None))
                lane_id = near_vehicle_info.get(VAR_LANE_INDEX, None)

                # Calculate lane difference between ego and nearby vehicle
                lane_diff = ego_lane_id - lane_id

                if lane_diff == 1:
                    right_availability = 0  # Right lane has a vehicle
                elif lane_diff == -1:
                    left_availability = 0   # Left lane has a vehicle
                elif lane_diff == 0:
                    can_change_lane = 1     # Same lane, should consider lane change

        # Construct the observation
        # Observation: (ego_position, ego_lane_id, ego_velocity, left_lane_availability, right_lane_availability, can_change_lane)
        observation = [
            # ego_state[0],          # Ego position
            ego_lane_id,           
            ego_state[2],          # Ego speed
            left_availability,     
            right_availability,    
            can_change_lane        
        ]
        #observation = np.array(observation).reshape(1,6)
        #print(f"obser:{observation}")
        return observation
        

    def simulate_collision(self):
        """
        Simulate a collision and stop it.
        """
        if self.show_gui:
        # Change vehicle color to red to indicate a collision
            traci.vehicle.setColor(self.egoID, (255, 0, 0))
            # Set vehicle speed to zero
            traci.vehicle.setSpeed(self.egoID, 0)
            print("Collision occurred!")
            # time.sleep(1)
            # self.reset()
        else:
            print("Collision occurred!")
            # time.sleep(1)
            # self.reset()

    def step(self, action):
        """
        Make a step on the agent.
        """
        #self.step_count += 1
        #invalid = False
        #reward = 0
        #done = bool(self.done)
        # info = {
        #     "step_count": self.step_count
        # }
        done = False
        # Get current observation
        observation = self.get_observation()
        # current_lane = observation[1]
        # current_speed = observation[2]
        # left_availability = observation[3]
        # right_availability = observation[4]       
        # can_change_lane = observation[5]
        current_lane = observation[0]
        current_speed = observation[1]
        left_availability = observation[2]
        right_availability = observation[3]       
        can_change_lane = observation[4]

        # Disable default vehicle speed and lane change modes
        all_vehicles = traci.vehicle.getIDList()
        for vehicle in all_vehicles:
            traci.vehicle.setSpeedMode(vehicle, 0)         # Disable speed mode
            traci.vehicle.setLaneChangeMode(vehicle, 0)    # Disable lane change mode

        # Action 0: Change left lane
        if action == 0:
            if current_lane == 3 or left_availability == 0:
                # Beyond rightmost lane, simulate collision
                print(f"coliision action: {action}")
                self.simulate_collision()
                done = True
                reward = -50
                #return invalid, observation, reward, done, info
                return observation, reward, done
            elif can_change_lane:
                # Necessary lane change to the right
                traci.vehicle.changeLane(self.egoID, current_lane + 1, self.lane_change_duration)
                reward = 50
            else:
                # Unnecessary lane change to the right
                traci.vehicle.changeLane(self.egoID, current_lane + 1, self.lane_change_duration)
                reward = -2

        # Action 1: Change right lane
        elif action == 1:
            if current_lane == 0 or right_availability == 0:
                # Beyond leftmost lane, simulate collision
                print(f"coliision action: {action}")
                self.simulate_collision()
                done = True
                reward = -50
                #return invalid, observation, reward, done, info
                return observation, reward, done
            elif can_change_lane:
                # Necessary lane change to the left
                traci.vehicle.changeLane(self.egoID, current_lane - 1, self.lane_change_duration)
                reward = 50
            else:
                # Unnecessary lane change to the left
                traci.vehicle.changeLane(self.egoID, current_lane - 1, self.lane_change_duration)
                reward = -2

        # Action 2: Idle
        elif action == 2:
            if can_change_lane:
                # Needed to change lane but did not, simulate collision
                print(f"coliision action: {action}")
                self.simulate_collision()
                done = True
                reward = -50
                return observation, reward, done
            if current_speed == self.maximum_speed:
                reward = 2
            else:
                reward = -5

        # Action 3: Accelerate
        elif action == 3:
            if can_change_lane:
                # Needed to change lane but did not, simulate collision
                print(f"coliision action: {action}")
                self.simulate_collision()
                done = True
                reward = -50
                return observation, reward, done
            if current_speed == self.maximum_speed:
                reward = 0
            else:
                # Increase speed within maximum speed limit
                new_speed = min(current_speed + self.acceleration, self.maximum_speed)
                #speed_diff = self.maximum_speed - new_speed
                traci.vehicle.setSpeed(self.egoID, new_speed)
                #reward = speed_diff * 5
                reward = 5

        # Action 4: Decelerate
        elif action == 4:
            if can_change_lane == 1 and left_availability == 0 and right_availability == 0:
                # Have no choice but decelerate
                # print(f"coliision action: {action}")
                # self.simulate_collision()
                # done = True
                # reward -= 50
                new_speed = max(current_speed - self.deceleration, self.minimum_speed)
                #speed_diff = self.maximum_speed - new_speed
                traci.vehicle.setSpeed(self.egoID, new_speed)
                reward = 20
            else:
                #Not encourage to decelerate
                new_speed = max(current_speed - self.deceleration, self.minimum_speed)
                speed_diff = self.maximum_speed - new_speed
                traci.vehicle.setSpeed(self.egoID, new_speed)
                reward = -5
                # Decrease speed within minimum speed limit
                # new_speed = max(current_speed - self.deceleration, self.minimum_speed)
                # speed_diff = self.maximum_speed - new_speed
                # traci.vehicle.setSpeed(self.egoID, new_speed)
                # reward += speed_diff * 0.5

        # if current_speed == self.maximum_speed:
        #         reward += 2
        # Velocity reward: the closer to maximum speed, the better
        #reward -= 1 * (self.maximum_speed - traci.vehicle.getSpeed(self.egoID))

        # Advance the simulation by one step
        traci.simulationStep()
        time.sleep(0.2)

        # Get the new observation after the simulation step
        new_observation = self.get_observation()
        return new_observation, reward, done

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        
        """
        self.step_count = 0
        self.done = False

        # Start SUMO with or without GUI based on the order
        if self.show_gui:
            sumo_path = checkBinary('sumo-gui')
        else:
            sumo_path = checkBinary('sumo')

        traci.start([sumo_path, "-c", self.sumo_config])
        
        # Disable default vehicle speed and lane change modes
        all_vehicles = traci.vehicle.getIDList()
        for vehicle in all_vehicles:
            traci.vehicle.setSpeedMode(vehicle, 0)        
            traci.vehicle.setLaneChangeMode(vehicle, 0)    

        # Advance the simulation by one step
        traci.simulationStep()
        time.sleep(0.2)

        # Return the initial observation
        return self.get_observation()

    def close(self):
        try:
            traci.close()
        except Exception as e:
            print(f"Error closing TraCI: {e}")
        time.sleep(0.2)
