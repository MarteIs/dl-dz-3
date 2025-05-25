import torch

def compute_recall(targets: torch.Tensor, preds: torch.Tensor, k: int) -> float:
    top_preds = preds[:, :k]
    expanded_targets = targets.unsqueeze(-1).expand_as(top_preds)
    correct = (top_preds == expanded_targets).any(dim=1)
    return correct.float().mean().item()

def compute_mrr(targets: torch.Tensor, preds: torch.Tensor) -> float:
    expanded_targets = targets.unsqueeze(1).expand_as(preds)
    correct = (expanded_targets == preds)
    ranks = torch.zeros_like(targets, dtype=torch.float)

    for i in range(correct.size(0)):
        matches = torch.where(correct[i])[0]
        if len(matches) > 0:
            ranks[i] = 1.0 / (matches[0].item() + 1)

    return ranks.mean().item()