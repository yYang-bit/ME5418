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

if __name__ == '__main__':
    env = SumoGym()
    env.reset()
    done = False
    reward = 0
    for step in range(100):
        if step ==7:
            result1 = env.step(1)
            print(f"{result1}")
        if step == 50:
            result3 = env.step(0)
            print(f"{result3}")
        else:
            time.sleep(0.25)
            result2 = env.step(3)
            print(f"{result2}")