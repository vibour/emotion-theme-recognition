"""Metrics"""
from typing import Dict, List, Tuple

import numpy as np
import sklearn.metrics
import torch
from torch import Tensor

import ml.losses


def extract_first_metric(metrics: Dict[str, np.ndarray]) -> float:
    return float(list(metrics.values())[0].mean())


def _avg_sigmoid(outs: Tensor) -> Tensor:
    return torch.mean(torch.nn.Sigmoid()(outs), 2)


def averaged_sigmoid(outputs: List[Tensor],
                     targets: List[Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    probas = torch.cat(list(map(_avg_sigmoid, outputs))).cpu().numpy()
    targs = torch.cat(targets).cpu().numpy()
    return probas, targs


def roc_auc(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    return sklearn.metrics.roc_auc_score(targets > 0.5, probas, average=None)


def pr_auc(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    return sklearn.metrics.average_precision_score(targets > 0.5,
                                                   probas,
                                                   average=None)


def bce_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = torch.nn.BCELoss(reduction='none')
    targs = torch.Tensor(targets)
    probs = torch.Tensor(probas)
    return torch.mean(loss(probs, targs), 0).cpu().numpy()


def focal_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = ml.losses.BinaryFocalLossWithLogits(reduction='none')
    targs = torch.Tensor(targets)
    outs = torch.logit(torch.Tensor(probas))
    return torch.mean(loss(outs, targs), 0).cpu().numpy()


def weighted_bce_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = ml.losses.WeightedBCEWithLogitsLoss(reduction="none")
    targs = torch.Tensor(targets)
    outs = torch.logit(torch.Tensor(probas))
    return torch.mean(loss(outs, targs), 0).cpu().numpy()
