import csv

import numpy as np
import torch
from stable_baselines3 import PPO

from PPO_new import SumoEnv
from sumoClass import SumoClass
from tlsClass import TLSClass

# Load trained PPO model
ppo_model = PPO.load("ppo_traffic_policy")

# Load trained DQN model
dqn_model = TLSClass()
dqn_model.load("trained_kernel_initialiser_2.pth")

def evaluate_policy(policy, env, episodes=1, is_dqn=False):
    results = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_waiting_time = 0
        total_queue_length = 0
        total_co2_emission = 0
        total_avg_speed = 0
        steps = 0

        while not done:
            if is_dqn:
                action = policy(state)  # DQN dùng act(state)
            else:
                action, _ = policy.predict(state)  # PPO dùng predict()

            state, reward, done, _ = env.step(action)
            total_reward += reward
            total_waiting_time += state[2]
            total_queue_length += state[3]
            total_co2_emission += state[1]
            total_avg_speed += state[0]
            steps += 1

        results.append({
            "episode": episode,
            "total_reward": total_reward,
            "avg_waiting_time": total_waiting_time / steps,
            "avg_queue_length": total_queue_length / steps,
            "avg_co2_emission": total_co2_emission / steps,
            "avg_speed": total_avg_speed / steps
        })

    return results

# Create SUMO environment
env = SumoEnv()

def dqn_policy(state):
    if isinstance(state, np.ndarray):
        state = state.tolist()
    elif not isinstance(state, list):
        state = [state]
    
    # DQN yêu cầu state gồm 3 phần: [image1, image2, vector]
    fake_image = np.zeros((12, 12, 1), dtype=np.float32)  # Giả lập ảnh 12x12x1
    state_vector = np.array(state, dtype=np.float32).reshape(-1)  # Chuyển thành vector 1D
    state_formatted = [fake_image, fake_image, state_vector]  # Tạo đầu vào đúng dạng
    
    return dqn_model.act(state_formatted)


# Evaluate PPO and DQN
dqn_results = evaluate_policy(dqn_policy, env, is_dqn=True)
ppo_results = evaluate_policy(ppo_model, env, is_dqn=False)

# Save results to CSV
with open("evaluation_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["episode", "total_reward", "avg_waiting_time", "avg_queue_length", "avg_co2_emission", "avg_speed"])
    writer.writeheader()
    writer.writerows([{**r, "model": "DQN"} for r in dqn_results])
    writer.writerows([{**r, "model": "PPO"} for r in ppo_results])

print("✅ Evaluation completed. Results saved in 'evaluation_results.csv'.")
