import os
import sys
from env import SumoGym
import traci
import time

# check if SUMO_HOME exists in environment variable
# if not, then need to declare the variable before proceeding
# makes it OS-agnostic
if "SUMO_HOME" in os.environ:
   tools = os.path.join(os.environ["SUMO_HOME"], "tools")
   sys.path.append(tools)
else:
   sys.exit("please declare environment variable 'SUMO_HOME'")

"""
This demo showcases the behavior of the ego car as it accelerates to its maximum speed. At step 7, 
the ego car performs a right lane change, and at step 50, it executes a left lane change.
"""

if __name__ == '__main__':
    env = SumoGym()
    env.reset()
    done = False
    reward = 0
    for step in range(100):
        if step ==7:
            env.step(1)
        if step == 50:
            env.step(0)
        else:
            time.sleep(0.25)
            env.step(3)