import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelLoss(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight


    def forward(self, codebook_loss, inputs, reconstructions,
                global_step, split="train"):
        
        dice = dice_loss(reconstructions, inputs)
        ce = binary_cross_entropy_loss(reconstructions, inputs)

        loss = dice + ce + self.codebook_weight * codebook_loss.mean()

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/dice_loss".format(split): dice.detach().mean(),
                "{}/ce_loss".format(split): ce.detach().mean(),
                }
        return loss, log
    

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算 Dice 损失。
    :param pred: 预测的分割掩码，形状为 (B, 1, H, W)。
    :param target: 真实的分割掩码，形状为 (B, 1, H, W)。
    :param smooth: 平滑项，防止分母为零。
    :return: Dice 损失。
    """
    pred = torch.sigmoid(pred)  # 将 logits 转换为概率
    pred = pred.view(pred.size(0), -1)  # 展平
    target = target.view(target.size(0), -1)  # 展平

    intersection = (pred * target).sum(dim=1)  # 交集
    union = pred.sum(dim=1) + target.sum(dim=1)  # 并集

    dice = (2.0 * intersection + smooth) / (union + smooth)  # Dice 系数
    return 1.0 - dice.mean()  # Dice 损失

def binary_cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算二值交叉熵损失。
    :param pred: 预测的 logits，形状为 (B, 1, H, W)。
    :param target: 真实的分割掩码，形状为 (B, 1, H, W)，值为 0 或 1。
    :return: 二值交叉熵损失。
    """
    # 将 logits 转换为概率值
    pred = torch.sigmoid(pred)
    # 计算二值交叉熵损失
    loss = F.binary_cross_entropy(pred, target, reduction='mean')
    return loss
