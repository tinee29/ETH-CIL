import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None, reduction="mean"):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.pos_weight is not None:
            # Compute weight tensor based on positive class weight
            weights = self.pos_weight * targets + (1 - targets)
        else:
            weights = torch.ones_like(targets)

        return F.binary_cross_entropy(inputs, targets, weight=weights, reduction=self.reduction)

    def __str__(self):
        return "WeightedBCELoss - pos_weight: {}, reduction: {}".format(
            self.pos_weight, self.reduction
        )


class ReachabilityLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ReachabilityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure kernel is on the same device as inputs
        binary_image = inputs.float()
        # pad with 1s on the borders
        binary_image = F.pad(binary_image, (1, 1, 1, 1), mode="constant", value=1)

        # Define horizontal and vertical filters
        horizontal_filter = torch.tensor([[-1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        vertical_filter = torch.tensor([[-1], [1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Move filters to the same device as the binary image
        horizontal_filter = horizontal_filter.to(binary_image.device)
        vertical_filter = vertical_filter.to(binary_image.device)

        # Apply convolution
        # Make sure binary_image has shape [batch_size, 1, height, width]
        if binary_image.dim() == 5:
            binary_image = binary_image.squeeze(1)
        elif binary_image.dim() == 3:
            binary_image = binary_image.unsqueeze(1)

        horizontal_grad = F.conv2d(binary_image, horizontal_filter)
        vertical_grad = F.conv2d(binary_image, vertical_filter)

        # Apply a threshold
        horizontal_disconnections = (horizontal_grad.abs() > 0).float()
        vertical_disconnections = (vertical_grad.abs() > 0).float()

        # Sum disconnections to get an approximation of connected components
        num_disconnections = horizontal_disconnections.sum() + vertical_disconnections.sum()

        # Define the loss as the number of disconnections (this is proportional to the number of connected components)
        loss = num_disconnections

        total_loss = loss / 2000000. + F.binary_cross_entropy(inputs, targets, reduction=self.reduction)

        return total_loss