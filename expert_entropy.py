import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 📌 1. Đọc dữ liệu
df = pd.read_csv("simulation_summary.csv")

# Chọn các cột cần chuẩn hóa
columns_to_normalize = ["average_speed", "co2_emission", "waiting_vehicles", "queue_length"]

# 📌 2. Chuẩn hóa bằng Min-Max Scaling
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# 📌 3. Log Transformation cho queue_length để giảm độ lệch
df["queue_length"] = np.log1p(df["queue_length"])  # log(1 + x) để tránh log(0)

# 📌 4. Tính trọng số bằng Entropy-Based Weighting
def entropy_weighting(data):
    P = data / data.sum(axis=0)  # Ma trận xác suất
    log_P = np.log1p(P)  # log(1 + p) để tránh log(0)
    entropy = (-1 / np.log(len(data))) * np.sum(P * log_P, axis=0)
    weights = (1 - entropy) / (np.sum(1 - entropy))  # Chuẩn hóa trọng số
    return weights

# Áp dụng Entropy Weighting
weights = entropy_weighting(df[columns_to_normalize].values)

# 📌 5. Hiển thị trọng số tính được
weight_dict = dict(zip(columns_to_normalize, weights))
print("📊 Trọng số Entropy-Based Weighting:", weight_dict)

# 📌 6. Tính expert_score dựa trên trọng số
df["expert_score"] = df[columns_to_normalize].dot(weights)

# 📌 7. Đảo ngược expert_score để càng gần 1 thì càng tốt
df["expert_score"] = 1 - df["expert_score"]

plt.hist(df["expert_score"], bins=20)
plt.title("Expert Score Distribution")
plt.show()

# 📌 8. Chuyển đổi sang JSON theo định dạng yêu cầu
data = []
for _, row in df.iterrows():
    entry = {
        "state": {
            "step_time": row["step_time"],
            "mean_waiting_time": row["mean_waiting_time"],
            "mean_travel_time": row["mean_travel_time"],
            "density": row["density"],
            "average_speed": row["average_speed"],
            "co2_emission": row["co2_emission"],
            "waiting_vehicles": row["waiting_vehicles"],
            "queue_length": row["queue_length"],
        },
        "action": {
            "current_phase": row["current_phase"],
            "next_phase": row["next_phase"]
        },
        "expert_score": row["expert_score"]
    }
    data.append(entry)

# 📌 9. Lưu lại dữ liệu đã chuẩn hóa và tính toán xong
df.to_csv("final_weighted_data.csv", index=False)
with open("irl_data.json", "w") as f:
    json.dump(data, f, indent=4)

print("✅ File đã lưu: final_weighted_data.csv & irl_data.json")
