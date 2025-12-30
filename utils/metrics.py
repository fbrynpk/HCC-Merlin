import numpy as np
import torch

def bootstrap_f1(preds, labels, n_bootstraps=1000, seed=None):
    rng = np.random.RandomState(seed)
    bootstrapped_f1_scores = []

    for i in range(n_bootstraps):
        # Randomly sample indices with replacement
        indices = rng.randint(0, len(preds), len(preds))
        sampled_preds = preds[indices]
        sampled_labels = labels[indices]

        # Calculate F1 for the sampled dataset
        f1 = compute_f1(sampled_preds, sampled_labels)
        bootstrapped_f1_scores.append(f1)

    # Calculate the confidence interval from the bootstrapped scores
    lower = np.percentile(bootstrapped_f1_scores, 2.5)
    upper = np.percentile(bootstrapped_f1_scores, 97.5)

    return lower, upper, bootstrapped_f1_scores


def compute_f1(preds, labels):
    TP = ((preds == 1) & (labels == 1)).astype(float).sum().item()
    FP = ((preds == 1) & (labels == 0)).astype(float).sum().item()
    FN = ((preds == 0) & (labels == 1)).astype(float).sum().item()

    if TP + FP == 0 or TP + FN == 0:
        return 0.0  # Avoid division by zero

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    try:
        F1 = 2 * (precision * recall) / (precision + recall)
    except:
        F1 = 0.0
    return F1


def evaluate_predictions(probs, labels_in):
    preds = torch.tensor(probs)
    labels = torch.tensor(labels_in)
    accuracy = (preds == labels).float().mean().item()

    # compute Acc. Sens. Spec. PPV NPV F1
    TP = ((preds == 1) & (labels == 1)).float().sum().item()
    TN = ((preds == 0) & (labels == 0)).float().sum().item()
    FP = ((preds == 1) & (labels == 0)).float().sum().item()
    FN = ((preds == 0) & (labels == 1)).float().sum().item()

    try:
        sensitivity = TP / (TP + FN)
    except:
        sensitivity = 0.0
    try:
        specificity = TN / (TN + FP)
    except:
        specificity = 0.0
    try:
        PPV = TP / (TP + FP)
    except:
        PPV = 0.0
    try:
        NPV = TN / (TN + FN)
    except:
        NPV = 0.0
    try:
        F1 = 2 * (PPV * sensitivity) / (PPV + sensitivity)
    except:
        F1 = 0.0

    print(f"F1: {F1}")
    lower, upper, bootstrapped_f1_scores = bootstrap_f1(preds.numpy(), labels.numpy())
    print(f"F1 Lower: {lower}")
    print(f"F1 Upper: {upper}")
    print("")

    return F1, lower, upper, bootstrapped_f1_scores