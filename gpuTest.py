import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PYTORCH USING: ", device)

y = torch.rand(50000, 30).cuda(device)
x = torch.rand(50000, 30).cuda(device)
for _ in tqdm(range(100000)):
    x*y

y1 = torch.rand(50000, 30)
x1 = torch.rand(50000, 30)
for _ in tqdm(range(100000)):
    x1*y1