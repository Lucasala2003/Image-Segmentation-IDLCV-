import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = 1 - (2 * (y_pred * y_true).sum() + 1) / (y_pred.sum() + y_true.sum() + 1)
        return loss

# class DiceLoss(nn.Module):
#     def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
#         super().__init__()
#         self.smooth, self.eps = smooth, eps

#     def forward(self, y_pred, y_true):
#         p = torch.sigmoid(y_pred)              # confidenze ∈ [0,1]
#         p = p.view(p.size(0), -1)
#         t = y_true.view(y_true.size(0), -1).float()
#         inter = (p * t).sum(dim=1)
#         denom = p.sum(dim=1) + t.sum(dim=1)
#         dice = (2*inter + self.smooth) / (denom + self.smooth + self.eps)
#         return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        alpha = 0.25
        gamma = 2.0
        bce_loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        p_t = torch.exp(-bce_loss)
        loss = alpha * (1 - p_t) ** gamma * bce_loss
        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean", eps: float = 1e-7):
#         super().__init__()
#         self.gamma, self.alpha, self.reduction, self.eps = gamma, alpha, reduction, eps

#     def forward(self, y_pred, y_true):
#         y_true = y_true.float()
#         # BCE per-pixel con logits
#         bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
#         p = torch.sigmoid(y_pred)
#         p_t = y_true * p + (1 - y_true) * (1 - p)                 # per-pixel
#         alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
#         loss = alpha_t * (1 - p_t).clamp(min=self.eps).pow(self.gamma) * bce
#         if self.reduction == "mean":
#             return loss.mean()
#         if self.reduction == "sum":
#             return loss.sum()
#         return loss


class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        regularization = torch.sum(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])) + \
                         torch.sum(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]))
        return loss + 0.1*regularization

# class BCELoss_TotalVariation(nn.Module):
#     def __init__(self, tv_weight: float = 1e-3, reduction: str = "mean"):
#         super().__init__()
#         self.tv_weight, self.reduction = tv_weight, reduction
#         self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

#     def forward(self, y_pred, y_true):
#         bce = self.bce(y_pred, y_true.float())
#         p = torch.sigmoid(y_pred)
#         dh = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
#         dw = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
#         tv = (dh.mean() + dw.mean()) if self.reduction == "mean" else (dh.sum() + dw.sum())
#         return bce + self.tv_weight * tv
