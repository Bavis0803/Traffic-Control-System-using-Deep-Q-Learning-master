import pandas as pd

# 📌 1️⃣ Đọc file CSV
df = pd.read_csv("simulation_summary.csv")

# 📌 2️⃣ Tính giá trị MAX từ dữ liệu thực tế
MAX_WAITING_TIME = df["waiting_vehicles"].max()
MAX_QUEUE_LENGTH = df["queue_length"].max()
MAX_AVERAGE_SPEED = df["average_speed"].max()
MAX_CO2_EMISSION = df["co2_emission"].max()

print("✅ Giá trị MAX:")
print(f"MAX_WAITING_TIME = {MAX_WAITING_TIME}")
print(f"MAX_QUEUE_LENGTH = {MAX_QUEUE_LENGTH}")
print(f"MAX_AVERAGE_SPEED = {MAX_AVERAGE_SPEED}")
print(f"MAX_CO2_EMISSION = {MAX_CO2_EMISSION}")

# 📌 3️⃣ Tính trọng số (range-based weighting)
range_values = {
    "waiting_vehicles": MAX_WAITING_TIME - df["waiting_vehicles"].min(),
    "queue_length": MAX_QUEUE_LENGTH - df["queue_length"].min(),
    "average_speed": MAX_AVERAGE_SPEED - df["average_speed"].min(),
    "co2_emission": MAX_CO2_EMISSION - df["co2_emission"].min()
}

# Chuẩn hóa trọng số
total_range = sum(range_values.values())
weights = {key: range_values[key] / total_range for key in range_values}

# Ánh xạ trọng số vào alpha, beta, gamma, delta
alpha, beta, gamma, delta = weights["waiting_vehicles"], weights["queue_length"], weights["average_speed"], weights["co2_emission"]

print("\n✅ Trọng số tính toán:")
print(f"Alpha (Waiting Time): {alpha:.4f}")
print(f"Beta (Queue Length): {beta:.4f}")
print(f"Gamma (Speed): {gamma:.4f}")
print(f"Delta (CO2): {delta:.4f}")

# 📌 4️⃣ Chuẩn hóa dữ liệu về khoảng [0, 1]
df["waiting_vehicles_norm"] = df["waiting_vehicles"] / MAX_WAITING_TIME
df["queue_length_norm"] = df["queue_length"] / MAX_QUEUE_LENGTH
df["average_speed_norm"] = df["average_speed"] / MAX_AVERAGE_SPEED
df["co2_emission_norm"] = df["co2_emission"] / MAX_CO2_EMISSION

# 📌 5️⃣ Tính expert_score
df["expert_score"] = (
    alpha * (1 - df["waiting_vehicles_norm"]) +
    beta * (1 - df["queue_length_norm"]) +
    gamma * df["average_speed_norm"] +
    delta * (1 - df["co2_emission_norm"])
)

df["expert_score"] = df["expert_score"].clip(0, 1)  # Đảm bảo nằm trong khoảng [0, 1]

# 📌 6️⃣ Xuất file CSV mới
df.to_csv("data_with_expert_score.csv", index=False)
print("\n✅ Đã tính `expert_score` và lưu vào `data_with_expert_score.csv`")
