from torch import nn
import torch


class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives_list: torch.Tensor) -> torch.Tensor:
        batch_size = anchors.size(0)
        num_negatives = negatives_list.size(1)

        pos_dist = torch.norm(anchors - positives, p=2, dim=1, keepdim=True)
        pos_dist = pos_dist.expand(batch_size, num_negatives)
        neg_dist = torch.norm(anchors.unsqueeze(1) - negatives_list, p=2, dim=2)

        raw_triplet_loss = pos_dist - neg_dist + self.margin
        active_mask = raw_triplet_loss > 0

        # Batch-all should average only active triplets; including easy zeros weakens gradients.
        if active_mask.any():
            return raw_triplet_loss[active_mask].mean()

        return anchors.new_tensor(0.0)

        
