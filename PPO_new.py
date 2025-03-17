import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sumolib import checkBinary

from DeepMAXIRL import RewardNet  # Load reward model
from sumoClass import SumoClass
from tlsClass import TLSClass

# Load reward model
reward_model = RewardNet(state_dim=4, action_dim=8)
reward_model.load_state_dict(torch.load("reward_model.pth"))
reward_model.eval()

# SUMO Environment Wrapper
import gym


class SumoEnv(gym.Env):
    def __init__(self):
        self.sumo = SumoClass(nodes=4, edges=4, lanes=3)
        self.tls = TLSClass()
        self.action_space = gym.spaces.Discrete(8)  # 4 possible actions
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)  # [average_speed, co2_emission, waiting_vehicles, queue_length]

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self.sumo.generate_routefile()
        traci.start([checkBinary("sumo-gui"), "-c", "data/cross.sumocfg", "--start", "--quit-on-end"])
        state = self.get_state()
        return np.array(state, dtype=np.float32)

    def get_state(self):
        state_irl = self.sumo.get_state_irl()
        state_values = [
            state_irl.get("average_speed", 0),
            state_irl.get("co2_emission", 0),
            state_irl.get("waiting_vehicles", 0),
            state_irl.get("queue_length", 0)
        ]
        state_values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in state_values]  # Ensure all values are float
        return np.array(state_values, dtype=np.float32)

    def step(self, action):
        traci.simulationStep()
        next_state = self.get_state()

        # Kiểm tra action hợp lệ
        if action is None or not isinstance(action, (int, np.integer)):
            raise ValueError(f"Invalid action received: {action}")

        # Compute reward using RewardNet
        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([int(action)], dtype=torch.long)  # Đảm bảo action là int
        reward = reward_model(state_tensor, action_tensor).item()

        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done, {}


    def close(self):
        traci.close()

if __name__ == "__main__":
    env = DummyVecEnv([lambda: SumoEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    model.save("ppo_traffic_policy")
    print("✅ PPO model saved as 'ppo_traffic_policy'.")

