import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 📌 1. Load dữ liệu từ JSON
with open("irl_data.json", "r") as f:
    data = json.load(f)

# 📌 2. Chuẩn bị dữ liệu (Loại bỏ `step_time`)
states, actions, expert_scores = [], [], []
for entry in data:
    state_dict = entry["state"].copy()
    del state_dict["step_time"]  # Loại bỏ `step_time`

    states.append(list(state_dict.values()))
    actions.append(list(entry["action"].values()))
    expert_scores.append(entry["expert_score"])

# Chuyển thành tensor
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)
expert_scores = torch.tensor(expert_scores, dtype=torch.float32).view(-1, 1)

# 📌 3. Chuẩn hóa expert_score (0 → 1)
expert_scores = (expert_scores - expert_scores.min()) / (expert_scores.max() - expert_scores.min())

# 📌 4. Gộp state và action
input_data = torch.cat((states, actions), dim=1)

# 📌 5. Chia train/test (80% train - 20% test)
dataset = TensorDataset(input_data, expert_scores)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 📌 6. DataLoader để mini-batch training
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 📌 7. Mô hình học reward (MLP)
class RewardNet(nn.Module):
    def __init__(self, input_dim):
        super(RewardNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

reward_model = RewardNet(input_data.shape[1])

# 📌 8. Huấn luyện mô hình (có đánh giá test loss)
optimizer = optim.Adam(reward_model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = nn.MSELoss()

epochs = 500
train_losses, test_losses = [], []

for epoch in range(epochs):
    # 📌 8.1 Train
    reward_model.train()
    epoch_train_loss = 0
    for batch_states, batch_scores in train_loader:
        optimizer.zero_grad()
        predicted_rewards = reward_model(batch_states)
        loss = loss_fn(predicted_rewards, batch_scores)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))

    # 📌 8.2 Đánh giá trên test set
    reward_model.eval()
    epoch_test_loss = 0
    with torch.no_grad():
        for batch_states, batch_scores in test_loader:
            predicted_rewards = reward_model(batch_states)
            loss = loss_fn(predicted_rewards, batch_scores)
            epoch_test_loss += loss.item()

    test_losses.append(epoch_test_loss / len(test_loader))

    # 📌 8.3 In kết quả mỗi 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}")

# 📌 9. Vẽ đồ thị Train Loss vs Test Loss
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Test Loss")
plt.show()

# 📌 10. Lưu lại mô hình reward
torch.save(reward_model.state_dict(), "learned_reward.pth")
print("✅ Reward function đã được học và lưu vào learned_reward.pth")

# 📌 11. Hàm dự đoán reward từ state + action
def predict_reward(state, action):
    reward_model.eval()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    input_tensor = torch.cat((state_tensor, action_tensor), dim=1)
    
    with torch.no_grad():
        predicted_reward = reward_model(input_tensor).item()
    
    return predicted_reward

# 📌 12. Ví dụ dự đoán
sample_state = states[0].tolist()  # Lấy state đầu tiên
sample_action = actions[0].tolist()  # Lấy action đầu tiên
predicted_reward = predict_reward(sample_state, sample_action)
print(f"🔮 Predicted Reward: {predicted_reward:.4f}")
