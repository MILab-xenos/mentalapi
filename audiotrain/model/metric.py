# import torch


# def accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)


# def top_k_acc(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)

import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# 需要转为 CPU numpy 计算 sklearn 指标
def f1_macro(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1).cpu().numpy()
        target = target.cpu().numpy()
    return f1_score(target, pred, average='macro', zero_division=0)

def recall_macro(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1).cpu().numpy()
        target = target.cpu().numpy()
    return recall_score(target, pred, average='macro', zero_division=0)