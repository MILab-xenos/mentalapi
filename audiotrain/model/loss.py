import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    """
    Standard CrossEntropyLoss
    output: (Batch_Size, Num_Classes) - Logits (未经过 Softmax 的原始输出)
    target: (Batch_Size) - Class Indices (0 or 1)
    """
    return F.cross_entropy(output, target)