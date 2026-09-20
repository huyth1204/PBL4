from src.ai.diffusion import build_diffusion
import torch
model = build_diffusion(53, hidden=256, T=100)
model.load_state_dict(torch.load("checkpoints_diffusion.pt"))
model.eval()