### Key Features

- **Observation Space**: The agent receives information about its current state, including its position, lane index, speed, and the availability of adjacent lanes.
- **Action Space**: The agent can choose to change lanes, accelerate, decelerate, or remain idle.


## Prerequisites

To run the simulations and obtain results, you must download and install the SUMO

**Update Path**: After installing SUMO, you need to update the path in the code to point to the correct location of your SUMO installation. Locate the following line in the `SumoGym` class:

   self.sumo_config = "/path/to/your/sumo/config/ottoman.sumocfg"