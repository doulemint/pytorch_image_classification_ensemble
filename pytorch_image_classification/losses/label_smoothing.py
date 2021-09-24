import torch
import torch.nn.functional as F
import yacs.config


def onehot_encoding(label: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)


def cross_entropy_loss(data: torch.Tensor, target: torch.Tensor,
                       reduction: str) -> torch.Tensor:
    logp = F.log_softmax(data, dim=1)
    # row_wise_max,_ = torch.max(logp,dim=1)
    # final_sum=torch.mean(row_wise_max)
    # c_tau = 0.8 * final_sum假如target_gt=1 logp_gt=0.8 则 loss为-0.8，但其他都为负
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

class cross_entropy_with_soft_target:
    def __init__(self,reduction: str):
        self.reduction = reduction

    def __call__(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        # print(targets.shape)
        # print(predictions.shape)
        # return torch.mean(torch.sum(-targets + F.log_softmax(predictions, dim=1), 1))
        return cross_entropy_loss(predictions, targets, self.reduction)


class LabelSmoothingLoss:
    def __init__(self, config: yacs.config.CfgNode, reduction: str):
        self.n_classes = config.dataset.n_classes
        self.epsilon = config.augmentation.label_smoothing.epsilon
        self.reduction = reduction

    def __call__(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        device = predictions.device

        onehot = onehot_encoding(
            targets, self.n_classes).type_as(predictions).to(device)
        targets = onehot * (1 - self.epsilon) + torch.ones_like(onehot).to(
            device) * self.epsilon / self.n_classes
        loss = cross_entropy_loss(predictions, targets, self.reduction)
        return loss

