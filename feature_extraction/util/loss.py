from torch import nn, optim
import torch


class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, anchors, positives, negatives_list):

        batch_size = anchors.size(0)
        num_negatives = negatives_list.size(
            1
        )


        #Eucledian dist
        pos_dist = torch.norm(
            anchors - positives, p=2, dim=1, keepdim=True
        )  # (batch_size, 1)
        pos_dist = pos_dist.expand(batch_size, num_negatives)
        neg_dist = torch.norm(anchors.unsqueeze(1) - negatives_list, p=2, dim=2)


        #Cosine sim
        # pos_dist = torch.sum(
        #     anchors * positives, dim=1, keepdim=True
        # )
        # pos_dist = pos_dist.expand(batch_size, num_negatives)
        # neg_dist = torch.sum(anchors.unsqueeze(1) * negatives_list, dim=2)



        triplet_loss = pos_dist - neg_dist + self.margin  # (batch_size, num_negatives)
        # triplet_loss = self.margin - (pos_dist - neg_dist)
         
        triplet_loss = torch.clamp(triplet_loss, min=0.0)  # Ensure non-negative loss

        if triplet_loss.numel() == 0:
            return torch.tensor(0.0, device=anchors.device, requires_grad=False)

        mean_loss = triplet_loss.mean()
        if not mean_loss.requires_grad:
            mean_loss = mean_loss.requires_grad_()
        # Return mean of valid triplet losses
        return mean_loss

        