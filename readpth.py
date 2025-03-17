import torch

# Đường dẫn đến file .pth
file_path = 'learned_reward.pth'

# Đọc file .pth
data = torch.load(file_path)

# In ra nội dung của file
print(data)