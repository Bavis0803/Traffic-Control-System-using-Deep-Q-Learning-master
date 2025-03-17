import numpy as np
import torch
import traci
from stable_baselines3 import PPO
from sumolib import checkBinary

from DeepMAXIRL import RewardNet  # Load reward model
from sumoClass import SumoClass
from tlsClass import TLSClass

# Load trained reward model
reward_model = RewardNet(state_dim=4, action_dim=8)
reward_model.load_state_dict(torch.load("reward_model.pth"))
reward_model.eval()

# Load trained PPO model
model = PPO.load("ppo_traffic_policy.zip")

# SUMO Environment Wrapper
import gym


class SumoEnv(gym.Env):
    def __init__(self):
        self.sumo = SumoClass(nodes=4, edges=4, lanes=3)
        self.tls = TLSClass()
        self.action_space = gym.spaces.Discrete(8)  # 4 possible actions
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self.sumo.generate_routefile()
        traci.start([checkBinary("sumo-gui"), "-c", "data/cross.sumocfg", "--start", "--quit-on-end"])
        return np.array(self.get_state(), dtype=np.float32)

    def get_state(self):
        state_irl = self.sumo.get_state_irl()
        state_values = [
            state_irl.get("average_speed", 0),
            state_irl.get("co2_emission", 0),
            state_irl.get("waiting_vehicles", 0),
            state_irl.get("queue_length", 0)
        ]
        return np.array([float(v) for v in state_values], dtype=np.float32)

    def step(self, action):
        traci.simulationStep()
        next_state = self.get_state()

        # Compute reward using RewardNet
        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([int(action)], dtype=torch.long)
        reward = reward_model(state_tensor, action_tensor).item()

        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done, {}

    def close(self):
        traci.close()


# Evaluate PPO Model Performance
def evaluate_model(env, model, episodes=5 ):
    total_rewards = []

    avg_co2_emissions = []
    avg_speeds = []
    avg_waiting_times = []
    avg_queue_lengths = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        co2_emissions =[]
        waiting_times = []
        speeds = []
        queue_lengths = []

        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)

            # Collect statistics
            total_reward += reward
            speeds.append(state[0])
            waiting_times.append(state[2])
            queue_lengths.append(state[3])
            co2_emissions.append(state[1])
        total_rewards.append(total_reward)
        avg_co2_emissions.append(np.mean(co2_emissions))
        avg_speeds.append(np.mean(speeds))
        avg_waiting_times.append(np.mean(waiting_times))
        avg_queue_lengths.append(np.mean(queue_lengths))

    print("\n📊 **Performance Evaluation**")
    
    print(f"✅ Average Reward: {np.mean(total_rewards):.2f}")
    print(f"average co2_emission: {np.mean(avg_co2_emissions):.2f}")
    print(f"🚗 Average Speed: {np.mean(avg_speeds):.2f}")
    print(f"⏳ Average Waiting Time: {np.mean(avg_waiting_times):.2f}")
    print(f"🚦 Average Queue Length: {np.mean(avg_queue_lengths):.2f}")


if __name__ == "__main__":
    env = SumoEnv()
    evaluate_model(env, model)
    env.close()