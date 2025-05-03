# import torch
# import torch.nn as nn
# from mamba_ssm import Mamba2
# from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


# class Verificator(nn.Module):
#     def __init__(self, d_model: int, n_mamba_layers: int = 2):
#         super().__init__()

#         self.search_proj = nn.Linear(1, d_model)

#         self.pos_embed = Summer(PositionalEncoding1D(d_model))

#         self.mamba_layers = nn.ModuleList(
#             [
#                 Mamba2(
#                     d_model=d_model,
#                     d_state=256,
#                     d_conv=4,
#                     expand=4,
#                     headdim=32,
#                 )
#                 for _ in range(n_mamba_layers)
#             ]
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, d_model // 2),
#             nn.GELU(),
#             nn.Linear(d_model // 2, 2),
#         )

#     def forward(
#         self,
#         db: torch.Tensor,
#         x: torch.Tensor,
#         search: torch.Tensor,
#     ):
#         search_embed = self.search_proj(search)
#         db = db + search_embed

#         z = torch.cat([db, x], dim=1)
#         z = self.pos_embed(z)

#         for layer in self.mamba_layers:
#             z = layer(z)

#         logits = self.classifier(z[:, -1, :])
#         return logits

# if __name__ == "__main__":
#     B, N, D = 4, 8, 128
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     db      = torch.randn(B, N, D, device=device)
#     x       = torch.randn(B, 1, D, device=device)
#     search  = torch.tensor([[[ 60.0]], [[250.0]], [[100.0]], [[290.0]]], device=device).expand(B, N, 1)

#     model = Verificator(d_model=D).to(device).eval()

#     with torch.no_grad():
#         logits = model(db, x, search)
#         probs  = logits.softmax(dim=-1)
#         pred   = probs.argmax(dim=-1)

#     print("logits:\n", logits)
#     print("probs :\n", probs)
#     print("pred  :\n", pred)
