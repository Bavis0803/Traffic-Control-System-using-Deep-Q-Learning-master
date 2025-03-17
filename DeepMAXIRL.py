import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# =========================
# 1. Định nghĩa Dataset
# =========================
class ExpertDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Bỏ qua dòng tiêu đề
            for row in reader:
                step_time, average_speed, co2_emission, waiting_vehicles, queue_length, \
                t_current_phase, t_next_phase, action, reward = map(float, row)
                
                # Định nghĩa state từ dữ liệu có sẵn
                state = [average_speed, co2_emission, waiting_vehicles, queue_length]

                action = int(action)
                reward = float(reward)
                next_phase = int(t_next_phase)

                self.data.append((state, action, reward, next_phase))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, reward, next_phase = self.data[idx]

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_phase = torch.tensor(next_phase, dtype=torch.long)

        return state, action, reward, next_phase

# =========================
# 2. Định nghĩa mạng RewardNet
# =========================
class RewardNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_action = nn.Linear(action_dim, 64)
        self.fc3 = nn.Linear(64 + 64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output reward

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # Chuyển action thành one-hot vector
        action_onehot = torch.nn.functional.one_hot(action, num_classes=8).float()
        a = torch.relu(self.fc_action(action_onehot))

        combined = torch.cat([x, a], dim=1)
        x = torch.relu(self.fc3(combined))
        reward_pred = self.fc4(x)
        return reward_pred

# =========================
# 3. Hàm huấn luyện
# =========================
def train_irl(model, dataloader, optimizer, epochs=10):
    model.train()
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for state, action, reward, _ in dataloader:
            optimizer.zero_grad()
            reward_pred = model(state, action)
            loss = loss_fn(reward_pred, reward.unsqueeze(1))  # MSE Loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# =========================
# 4. Huấn luyện mô hình
# =========================
dataset = ExpertDataset('expert_trajectories_irl.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Khởi tạo mô hình
state_dim = 4  # Số lượng features trong state
action_dim = 8  # 4 actions
model = RewardNet(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
train_irl(model, dataloader, optimizer, epochs=10)

# Lưu mô hình
torch.save(model.state_dict(), 'reward_model.pth')
print("✅ Model saved to 'reward_model.pth'.")
