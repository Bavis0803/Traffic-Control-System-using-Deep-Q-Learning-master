import csv

import numpy as np
import traci
from sumolib import checkBinary

from sumoClass import SumoClass
from tlsClass import TLSClass

# Đường dẫn mô hình DQN đã train
MODEL_PATH = 'trained_kernel_initialiser_2.pth'
NUM_EPISODES = 50

# Khởi tạo SUMO
sumoBinary = checkBinary('sumo-gui')
sumoObject = SumoClass(nodes=4, edges=4, lanes=3)
tlsObject = TLSClass()

# Load mô hình DQN và tắt khám phá
tlsObject.load(MODEL_PATH)
tlsObject.epsilon = 0.1

# Danh sách lưu trữ dữ liệu quỹ đạo
trajectory_data = []
raw_rewards = []

# Trọng số
weights = {
    'average_speed': 0.4,  # Tăng trọng số
    'co2_emission': -0.15,  # Giảm nhẹ trọng số âm
    'waiting_time': -0.35,  # Tăng trọng số âm
    'queue_length': -0.45   # Tăng trọng số âm
}

# Thu thập dữ liệu
for episode in range(NUM_EPISODES):
    print(f"Episode {episode + 1}/{NUM_EPISODES}: Đang thu thập dữ liệu...")
    sumoObject.generate_routefile()
    traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", "tripinfo.xml", "--quit-on-end"])
    
    step = 0
    last_phase = traci.trafficlight.getPhase("0")
    
    while traci.simulation.getMinExpectedNumber() > 0 and step < 10000:
        state_irl = sumoObject.get_state_irl()
        state = sumoObject.get_state()
        action = tlsObject.act(state)
        traci.simulationStep()
        current_phase = traci.trafficlight.getPhase("0")

        if current_phase != last_phase:
            average_speed = state_irl.get("average_speed", 0)
            co2_emission = state_irl.get("co2_emission", 0)
            waiting_vehicles = state_irl.get("waiting_vehicles", 0)
            queue_length = state_irl.get("queue_length", 0)

            raw_reward = (weights['average_speed'] * average_speed +
                          weights['co2_emission'] * co2_emission +
                          weights['waiting_time'] * waiting_vehicles +
                          weights['queue_length'] * queue_length)
            raw_rewards.append(raw_reward)

            trajectory_data.append({
                "step_time": step,
                "average_speed": round(average_speed, 2),
                "co2_emission": round(co2_emission, 2),
                "waiting_vehicles": round(waiting_vehicles, 2),
                "queue_length": queue_length,
                "t_current_phase": last_phase,
                "t_next_phase": current_phase,
                "action": action,
                "raw_reward": round(raw_reward, 2)
            })
            last_phase = current_phase
        
        step += 1
    
    traci.close()

# Chuẩn hóa reward bằng robust scaling
min_bound = np.percentile(raw_rewards, 5)
max_bound = np.percentile(raw_rewards, 95)
for data in trajectory_data:
    raw_reward = data["raw_reward"]
    normalized_reward = (raw_reward - min_bound) / (max_bound - min_bound) * 10
    data["reward"] = max(0, min(10, normalized_reward))

# Lưu dữ liệu vào CSV
with open('expert_trajectories_irl.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ["step_time", "average_speed", "co2_emission", "waiting_vehicles", "queue_length", "t_current_phase", "t_next_phase", "action", "reward"]
    writer.writerow(header)
    
    for data in trajectory_data:
        row = [data["step_time"], data["average_speed"], data["co2_emission"],
               data["waiting_vehicles"], data["queue_length"],
               data["t_current_phase"], data["t_next_phase"], data["action"],
               data["reward"]]
        writer.writerow(row)

print(f"Đã lưu {len(trajectory_data)} quỹ đạo vào 'expert_trajectories_irl.csv'.")
