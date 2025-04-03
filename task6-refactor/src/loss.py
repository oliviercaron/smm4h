# src/loss.py
import torch
import torch.nn.functional as F

def focal_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    alpha: float = 1.0, 
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal Loss implementation
    
    Args:
        logits (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
        alpha (float): Weighting factor for positive class
        gamma (float): Focusing parameter
    
    Returns:
        torch.Tensor: Computed focal loss
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss_val = alpha * ((1 - pt) ** gamma) * ce_loss
    return focal_loss_val.mean()

def dice_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Dice Loss implementation
    
    Args:
        logits (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
        smooth (float): Smoothing factor to prevent division by zero
    
    Returns:
        torch.Tensor: Computed dice loss
    """
    num_classes = logits.size(1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    probs = torch.softmax(logits, dim=1)
    
    intersection = torch.sum(probs * targets_one_hot, dim=0)
    union = torch.sum(probs, dim=0) + torch.sum(targets_one_hot, dim=0)
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()

def combined_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    alpha: float = 0.33, 
    beta: float = 0.33, 
    gamma: float = 0.33
) -> torch.Tensor:
    """
    Combined Loss function
    
    Args:
        logits (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
        alpha (float): Cross-entropy loss weight
        beta (float): Focal loss weight
        gamma (float): Dice loss weight
    
    Returns:
        torch.Tensor: Combined loss value
    """
    ce_loss = F.cross_entropy(logits, targets)
    fl = focal_loss(logits, targets)
    dl = dice_loss(logits, targets)
    
    combined_loss_val = alpha * ce_loss + beta * fl + gamma * dl
    return combined_loss_val