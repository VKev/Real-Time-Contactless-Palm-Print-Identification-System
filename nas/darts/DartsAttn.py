import torch
import torch.nn as nn
import torch.nn.functional as F
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention

class DARTSMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, candidate_heads):
        super(DARTSMultiHeadAttention, self).__init__()
        self.candidate_heads = candidate_heads
        self.num_candidates = len(candidate_heads)
        self.alpha = nn.Parameter(torch.randn(self.num_candidates))
        self.attn_modules = nn.ModuleList([
            ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=head)
            for head in candidate_heads
        ])
        self.discretized = False

    def discretize(self):
        best_index = torch.argmax(self.alpha).item()
        self.attn_modules = nn.ModuleList([self.attn_modules[best_index]])
        self.candidate_heads = [self.candidate_heads[best_index]]
        self.discretized = True
        with torch.no_grad():
            self.alpha.data.zero_()
            self.alpha.data[best_index] = 1.0

    def forward(self, q, k, v):
        if self.discretized:
            return self.attn_modules[0](q, k, v)
        else:
            weights = F.softmax(self.alpha, dim=0)
            out_weighted = 0
            for i, attn in enumerate(self.attn_modules):
                out_weighted += weights[i] * attn(q, k, v)
            return out_weighted

if __name__ == "__main__":
    candidate_heads = [2, 4, 8]

    darts_attn = DARTSMultiHeadAttention(
        d_model=256,
        d_k=256,
        d_v=256,
        candidate_heads=candidate_heads
    ).to('cuda')

    q = torch.randn(8, 205, 256).to('cuda')
    k = torch.randn(8, 205, 256).to('cuda')
    v = torch.randn(8, 205, 256).to('cuda')

    output_before = darts_attn(q, k, v)
    print("Output shape before discretization:", output_before.shape)

    print("Alphas before discretization:", darts_attn.alpha)
    print("Softmaxed alphas:", F.softmax(darts_attn.alpha, dim=0))

    print(darts_attn)
    darts_attn.discretize()
    output_after = darts_attn(q, k, v)
    print("Output shape after discretization:", output_after.shape)

    assert darts_attn.discretized, "The module has not been discretized!"

    print("Discretization successful. Selected candidate heads:", darts_attn.candidate_heads[0])
    print(darts_attn)