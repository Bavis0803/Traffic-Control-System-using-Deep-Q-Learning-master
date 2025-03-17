import csv

import numpy as np
import torch
import traci
from sumolib import checkBinary

from sumoClass import SumoClass  # Import class SumoClass


# 📌 1. Load mô hình đã huấn luyện
class RewardNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(RewardNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def load_reward_model(model_path, input_dim):
    """ Load mô hình IRL đã huấn luyện """
    model = RewardNet(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 📌 2. Kiểm nghiệm mô hình trên SUMO + Ghi hành động vào CSV
def test_performance(trained_model_path, output_csv="actions_irl.csv", num_episodes=1):
    sumoObject = SumoClass(nodes=4, edges=4, lanes=3)  # ✅ Truyền tham số đúng
    sumoBinary = checkBinary('sumo-gui')  # Chạy SUMO với giao diện

    # 📌 Load mô hình RewardNet
    reward_model = load_reward_model(trained_model_path, input_dim=9)  # 8 state + 1 action

    # 📌 Mở file CSV để lưu hành động
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "action"])  # Header

        # 📌 Khởi tạo thống kê hiệu suất
        total_reward = 0
        total_waiting_time = 0
        total_queue_length = 0
        total_co2_emission = 0

        for episode in range(num_episodes):
            print(f"🛠️ Testing episode {episode + 1}/{num_episodes}")
            stepz = 0
            traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])

            while traci.simulation.getMinExpectedNumber() > 0 and stepz < 10000:
                traci.simulationStep()
                
                # 📌 2.1 Lấy trạng thái từ SUMO
                state = sumoObject.get_state_irl()
                possible_actions = sumoObject.get_possible_actions()

                # 📌 2.2 Chuyển state thành tensor
                state_tensor = torch.tensor(list(state.values()), dtype=torch.float32).unsqueeze(0)

                # 📌 2.3 Tính reward cho mỗi action và chọn action tốt nhất
                best_action = None
                best_reward = -float('inf')

                for action in possible_actions:
                    action_tensor = torch.tensor([[action]], dtype=torch.float32)  # ✅ Đảm bảo tensor 2D
                    input_tensor = torch.cat((state_tensor, action_tensor), dim=1)  # Ghép state + action
                    reward = reward_model(input_tensor).item()  # Dự đoán reward

                    if reward > best_reward:
                        best_reward = reward
                        best_action = action

                # 📌 2.4 Thực hiện hành động tốt nhất
                if best_action is not None:
                    sumoObject.apply_action(best_action)  # ✅ Truyền action đã chọn
                    writer.writerow([stepz, best_action])  # Lưu thời gian + hành động vào CSV

                # 📌 2.5 Ghi nhận hiệu suất
                co2_emission = state["co2_emission"]
                waiting_time = state["waiting_vehicles"]
                queue_length = state["queue_length"]
                total_waiting_time += waiting_time
                total_queue_length += queue_length
                total_co2_emission += co2_emission
                total_reward += best_reward

                stepz += 1

            traci.close()

    # 📌 3. Tính trung bình các chỉ số hiệu suất
    avg_reward = total_reward / num_episodes
    avg_waiting_time = total_waiting_time / num_episodes
    avg_queue_length = total_queue_length / num_episodes
    avg_co2_emission = total_co2_emission / num_episodes

    print(f"✅ Average Reward: {avg_reward}")
    print(f"✅ Average Waiting Time: {avg_waiting_time}")
    print(f"✅ Average Queue Length: {avg_queue_length}")
    print(f"✅ Average CO2 Emission: {avg_co2_emission}")
    print(f"✅ Hành động đã lưu vào file: {output_csv}")

# 📌 4. Chạy kiểm nghiệm mô hình và lưu hành động
if __name__ == "__main__":
    trained_model_path = 'learned_reward.pth'  # Đường dẫn đến mô hình đã train
    test_performance(trained_model_path, output_csv="actions_irl.csv", num_episodes=1)