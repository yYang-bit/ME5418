## File Structure
- `A2C_model.py`: Defines the structure of the Actor-Critic model.
- `DQN_model.py`: Defines the DQN model and its agent class.
- `env.py`: Implements the SUMO environment, including state observation, action execution, and environment reset functionalities.
- `A2C_Agent.py`: Defines the training and decision-making logic for the A2C agent.
- `model_test.py`: Tests the model and agent's performance in the SUMO environment.

## File Descriptions

bash
.
├── sumo_env/
│ ├── ottoman.sumocfg # SUMO configuration file\n
│ ├── test05_path.net.xml # Road network definition
│ ├── test05_car.rou.xml # Vehicle route definition
│ └── test05_car.rou_1.xml # Alternative vehicle route definition
├── A2C_Agent.py # A2C agent implementation
├── A2C_model.py # A2C network model
├── env.py # Environment wrapper
└── Analysis.py # Data analysis and visualization


### 1. `A2C_model.py`
This file implements the Actor-Critic model, including shared layers, actor network, and critic network. The model's forward method returns action probabilities and state values.


### 2. `env.py`
This file implements the logic of the SUMO environment, including the definition of state and action spaces, vehicle state retrieval, environment reset, and stepping functionalities. The agent learns by interacting with the environment.

### 3. `A2C_Agent.py`
This file implements the logic for the A2C agent, including action selection and model training functionalities. The agent selects actions based on the current state and updates its policy through interaction with the environment.

### 4. `main.py`
This file is used to test the trained model and agent's performance in the SUMO environment. It initializes the environment and agent, running multiple steps to observe the agent's decision-making process.

## Usage Instructions
### 1. Create a Virtual Environment with Conda

If you don't have Conda installed, please install it first. Then, create a new Conda environment for this project:

```bash
conda create -n me5418 python=3.8
conda activate me5418
```

### 2. Install Python Dependencies

Install the required Python packages for the project:

```bash
pip install -r requirements.txt
```

### 3. Set Up SUMO Environment

Make sure you have SUMO installed on your machine. Set the `SUMO_HOME` environment variable to point to your SUMO installation directory:

```bash
export SUMO_HOME=/path/to/your/sumo
```
And change the adress in env.py

```bash
self.sumo_config = "/path/to/your/sumo/config/ottoman.sumocfg"
```

### 4. Create ROS workspace
```bash
mkdir -p /<YOUR_HOME_DIR>/me5418_ws/src
cd /<YOUR_HOME_DIR>/me5418_ws/src
```

### 5. Clone This Repository

Clone the repository containing the traffic simulation project:

```bash
git clone https://github.com/yYang-bit/ME5418.git
```

### 6. Build the workspace

```bash
source devel/setup.bash
catkin_make
```

### 7. Run the Project

```bash
python main.py
```