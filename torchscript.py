import os
import torch
from feature_extraction.model import MyModel

model = MyModel().cuda()
ckpt = torch.load("feature_extraction/checkpoints/attempt_5.pth")
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

example = torch.randn(1, 3, 224, 224).cuda()

traced = torch.jit.trace(model, example, check_trace=False)

model_name = "feature_extraction"
version = "1"
repo_root = "model_repository"
model_dir = os.path.join(repo_root, model_name)
version_dir = os.path.join(model_dir, version)

os.makedirs(version_dir, exist_ok=True)

traced_path = os.path.join(version_dir, "model.pt")
traced.save(traced_path)
print(f"Saved TorchScript model to {traced_path}")
