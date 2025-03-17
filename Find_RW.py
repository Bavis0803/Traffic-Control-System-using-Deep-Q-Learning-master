from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# 1. Đọc dữ liệu từ file CSV
# Đọc dữ liệu gốc (trajectories chuyên gia)
data = pd.read_csv('expert_trajectories_irl.csv')
# Đọc dữ liệu đã gắn nhãn phần thưởng
labeled_data = pd.read_csv('labeled_trajectories.csv')

# 2. Chuẩn hóa dữ liệu
scaler = StandardScaler()
states_raw = data[['average_speed', 'co2_emission', 'waiting_vehicles', 'queue_length', 't_current_phase', 't_next_phase']].values
states_normalized = scaler.fit_transform(states_raw)

states = [tuple(state) for state in states_normalized]
actions = data['action'].values
action_space = np.unique(actions).tolist()

# Lấy nhãn phần thưởng từ file labeled_trajectories.csv
rewards_labeled = labeled_data['reward'].values

# 3. Định nghĩa đặc trưng (thêm đặc trưng phi tuyến và chuẩn hóa)
def features(state, action):
    base_features = np.array(state)
    waiting_vehicles = base_features[2]
    queue_length = base_features[3]
    # Chuẩn hóa các đặc trưng phi tuyến để tránh giá trị quá lớn
    waiting_vehicles_squared = waiting_vehicles**2
    inv_queue_length = 1 / (queue_length + 1e-5)
    # Giới hạn giá trị để tránh bùng nổ
    waiting_vehicles_squared = np.clip(waiting_vehicles_squared, -100, 100)
    inv_queue_length = np.clip(inv_queue_length, -10, 10)
    
    extended_features = np.array([
        base_features[0],          # average_speed
        base_features[1],          # co2_emission
        waiting_vehicles,           # waiting_vehicles
        waiting_vehicles_squared,  # waiting_vehicles^2
        inv_queue_length,          # 1/queue_length
        base_features[4],          # t_current_phase
        base_features[5]           # t_next_phase
    ])
    return extended_features

# Cập nhật số chiều đặc trưng
feature_dim = 7  # 6 đặc trưng gốc + 1 đặc trưng phi tuyến

# 4. Tính xác suất chọn hành động
def compute_policy(theta, state, action_space):
    q_values = {}
    for action in action_space:
        phi = features(state, action)
        q_values[action] = np.dot(theta, phi)
    max_q = max(q_values.values())
    # Giới hạn giá trị exp để tránh bùng nổ
    exp_q = {a: np.exp(np.clip(q - max_q, -500, 500)) for a, q in q_values.items()}
    z = sum(exp_q.values())
    if z == 0:
        z = 1e-10
    policy = {a: exp_q[a] / z for a in action_space}
    return policy

# 5. Tính gradient
def compute_gradient(theta, trajectories, action_space, feature_dim):
    expected_features_expert = np.zeros(feature_dim)
    for state, action in trajectories:
        phi = features(state, action)
        expected_features_expert += phi
    expected_features_expert /= len(trajectories)

    expected_features_policy = np.zeros(feature_dim)
    for state, _ in trajectories:
        policy = compute_policy(theta, state, action_space)
        for action in action_space:
            phi = features(state, action)
            expected_features_policy += policy[action] * phi

    gradient = expected_features_expert - expected_features_policy
    return gradient

# 6. Hàm mục tiêu (thêm thành phần mất mát từ nhãn phần thưởng)
def objective(theta, trajectories, action_space, feature_dim, alpha, rewards_labeled, lambda_reg):
    # Thành phần log-likelihood từ MaxEnt IRL
    log_likelihood = 0
    for state, action in trajectories:
        policy = compute_policy(theta, state, action_space)
        if policy[action] <= 0:
            policy[action] = 1e-10
        log_likelihood += np.log(policy[action])
    
    # Thành phần điều chuẩn L2
    reg_term = alpha * np.sum(theta**2)

    # Thành phần mất mát từ nhãn phần thưởng
    reward_loss = 0
    for i, (state, action) in enumerate(trajectories):
        phi = features(state, action)
        predicted_reward = np.dot(theta, phi)
        labeled_reward = rewards_labeled[i]
        reward_loss += (predicted_reward - labeled_reward) ** 2
    reward_loss /= len(trajectories)

    # Tổng hàm mục tiêu
    total_loss = -log_likelihood + reg_term + lambda_reg * reward_loss
    return total_loss

# 7. Callback để ghi lại thông tin hội tụ
class OptimizationLogger:
    def __init__(self):
        self.iterations = []
        self.objective_values = []
        self.gradients = []
        self.thetas = []

    def callback(self, theta):
        obj_value = objective(theta, trajectories, action_space, feature_dim, alpha=1.0, rewards_labeled=rewards_labeled, lambda_reg=0.01)
        grad = compute_gradient(theta, trajectories, action_space, feature_dim)
        self.iterations.append(len(self.iterations) + 1)
        self.objective_values.append(obj_value)
        self.gradients.append(np.linalg.norm(grad))
        self.thetas.append(theta.copy())
        print(f"Lần lặp {self.iterations[-1]}: Objective = {obj_value:.6f}, Gradient norm = {self.gradients[-1]:.6f}")

# 8. Huấn luyện IRL
def maxent_irl(trajectories, action_space, feature_dim, rewards_labeled, max_iter=1000, tol=1e-6, alpha=1.0, lambda_reg=0.01):
    np.random.seed(42)
    theta = np.random.randn(feature_dim)
    
    logger = OptimizationLogger()
    result = minimize(
        fun=objective,
        x0=theta,
        args=(trajectories, action_space, feature_dim, alpha, rewards_labeled, lambda_reg),
        method='SLSQP',
        jac=lambda theta, *args: compute_gradient(theta, args[0], args[1], args[2]),
        constraints=[
            {'type': 'ineq', 'fun': lambda theta: -theta[2]},  # theta_3 <= 0 (phạt waiting_vehicles)
            {'type': 'ineq', 'fun': lambda theta: -theta[3]},  # theta_4 <= 0 (phạt waiting_vehicles^2)
            {'type': 'ineq', 'fun': lambda theta: theta[4]},   # theta_5 >= 0 (thưởng 1/queue_length)
            {'type': 'ineq', 'fun': lambda theta: theta[6]}    # theta_7 >= 0 (thưởng t_next_phase)
        ],
        callback=logger.callback,
        options={'maxiter': max_iter, 'disp': True}
    )

    print("\nKết quả tối ưu hóa:")
    print(f"Trạng thái hội tụ: {result.success}")
    print(f"Lý do dừng: {result.message}")
    print(f"Objective cuối cùng: {result.fun:.6f}")
    print(f"Gradient norm cuối cùng: {np.linalg.norm(result.jac):.6f}")
    
    if len(logger.thetas) >= 2:
        delta_theta = np.linalg.norm(logger.thetas[-1] - logger.thetas[-2])
        print(f"Thay đổi của theta ở lần lặp cuối: {delta_theta:.6f}")
    
    return result.x, logger

# 9. Main: Chuẩn bị dữ liệu và huấn luyện
trajectories = list(zip(states, actions))

theta, logger = maxent_irl(trajectories, action_space, feature_dim, rewards_labeled)

# 10. In kết quả
print("\nTrọng số của hàm phần thưởng (theta):")
print("average_speed:", theta[0])
print("co2_emission:", theta[1])
print("waiting_vehicles:", theta[2])
print("waiting_vehicles_squared:", theta[3])
print("inv_queue_length:", theta[4])
print("t_current_phase:", theta[5])
print("t_next_phase:", theta[6])

# 11. Vẽ đồ thị hội tụ
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(logger.iterations, logger.objective_values, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Objective Value (Negative Log-Likelihood + Reward Loss)')
plt.title('Convergence of Objective Value')

plt.subplot(1, 2, 2)
plt.plot(logger.iterations, logger.gradients, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
plt.title('Convergence of Gradient Norm')

plt.tight_layout()
plt.show()

# 12. Tính phần thưởng cho một số trạng thái
def reward(state, action, theta):
    phi = features(state, action)
    return np.dot(theta, phi)

for i in range(5):
    state, action = trajectories[i]
    r = reward(state, action, theta)
    print(f"Trạng thái {i}: {state}, Hành động: {action}, Phần thưởng: {r}")