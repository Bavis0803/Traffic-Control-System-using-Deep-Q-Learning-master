import pandas as pd

# Đọc dữ liệu từ file
data = pd.read_csv('expert_trajectories_irl.csv')

# Chuẩn bị trajectories
states = data[['average_speed', 'co2_emission', 'waiting_vehicles', 'queue_length', 't_current_phase', 't_next_phase']].values
actions = data['action'].values
trajectories = list(zip(states, actions))

# Gắn nhãn phần thưởng dựa trên đầy đủ các chỉ số
labeled_data = []
for state, action in trajectories:
    average_speed = state[0]
    co2_emission = state[1]
    waiting_vehicles = state[2]
    queue_length = state[3]
    
    # Trạng thái ùn tắc nặng: Phạt mạnh
    if waiting_vehicles > 200 and queue_length > 20 and average_speed < 10 and co2_emission > 150:
        reward = -10
    # Trạng thái ùn tắc nhẹ: Phạt vừa
    elif waiting_vehicles > 50 and queue_length > 10 and average_speed < 20:
        reward = -5
    # Trạng thái trơn tru: Thưởng
    elif average_speed > 30 and waiting_vehicles < 50 and queue_length < 10 and co2_emission < 50:
        reward = 5
    # Trạng thái trung bình
    else:
        reward = 0
    
    labeled_data.append((state, action, reward))

# Chuyển labeled_data thành DataFrame
labeled_df = pd.DataFrame(
    [(list(state), action, reward) for state, action, reward in labeled_data],
    columns=['state', 'action', 'reward']
)

# Tách cột state thành các cột riêng
labeled_df[['average_speed', 'co2_emission', 'waiting_vehicles', 'queue_length', 't_current_phase', 't_next_phase']] = pd.DataFrame(
    labeled_df['state'].tolist(), index=labeled_df.index
)
labeled_df = labeled_df.drop(columns=['state'])  # Xóa cột state tổng hợp

# Xuất ra file CSV
labeled_df.to_csv('labeled_trajectories.csv', index=False)

# Đếm số lượng mỗi loại phần thưởng
rewards = labeled_df['reward'].tolist()
reward_counts = {
    -10: rewards.count(-10),
    -5: rewards.count(-5),
    5: rewards.count(5),
    0: rewards.count(0)
}

print("Số lượng phần thưởng:")
print(f"reward = -10 (ùn tắc nặng: waiting_vehicles > 200, queue_length > 20, average_speed < 10, co2_emission > 150): {reward_counts[-10]}")
print(f"reward = -5 (ùn tắc nhẹ: waiting_vehicles > 50, queue_length > 10, average_speed < 20): {reward_counts[-5]}")
print(f"reward = 5 (trơn tru: average_speed > 30, waiting_vehicles < 50, queue_length < 10, co2_emission < 50): {reward_counts[5]}")
print(f"reward = 0 (trung bình): {reward_counts[0]}")