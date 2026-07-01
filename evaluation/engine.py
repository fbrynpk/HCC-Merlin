import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from evaluation.metrics import evaluate_predictions


def evaluate_model(model, val_loader, device, prompts):
    """
    Contrastive zero-shot evaluation using positive/negative text prompts.

    Args:
        model: Merlin model wrapper.
        val_loader: DataLoader yielding batches with 'image' and 'labels'.
        device: Torch device string.
        prompts: DiseasePrompts dataclass with positive_prompts and negative_prompts.

    Returns:
        (f1, lower, upper, recall, precision, accuracy, all_preds, all_labels)
    """
    model.eval()
    image_features_all = []
    all_labels = []
    all_preds = []
    all_probs = []
    num_positive = len(prompts.positive_prompts)

    with torch.no_grad():
        text_features = model.model.encode_text(
            prompts.positive_prompts + prompts.negative_prompts
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for batch in tqdm(val_loader, desc="Evaluating contrastive performance"):
            img_data = batch["image"].to(device)
            all_labels.extend(batch["labels"].tolist())
            feats = model.model.encode_image(img_data)[0]
            image_features_all.append(feats.cpu())

        image_features_all = torch.cat(image_features_all, dim=0)

        for img_feat in image_features_all:
            img_feat = img_feat.to(device)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            img_feat = img_feat.unsqueeze(0)

            similarity = 100.0 * img_feat @ text_features.T  # (1, P+N)
            similarity_positive = similarity[:, :num_positive]
            similarity_positive = similarity_positive.mean(
                dim=1, keepdim=True
            )  # (1, 1)

            similarity_negative = similarity[:, num_positive:]
            similarity_negative = similarity_negative.mean(
                dim=1, keepdim=True
            )  # (1, 1)

            similarities = torch.cat(
                [similarity_negative, similarity_positive], dim=1
            )  # (1, 2)

            all_preds.append(similarities.argmax(dim=1).item())
            all_probs.append(
                torch.sigmoid(similarity_positive - similarity_negative).item()
            )

    # with torch.no_grad():
    #     text_features = model.model.encode_text(
    #         prompts.positive_prompts + prompts.negative_prompts
    #     )

    #     for i, batch in tqdm(enumerate(val_loader), desc="Evaluating contrastive performance"):
    #         img_data = batch["image"].to(device)
    #         all_labels.append(batch["labels"].item())
    #         image_features = model.model.encode_image(img_data)[0]  # (1, D)

    #         for img_feature in image_features:
    #             image_features_all.append(img_feature.cpu().numpy())

    #     text_features /= text_features.norm(dim=-1, keepdim=True)

    #     for img_feature in image_features_all:
    #         img_feature = torch.tensor(img_feature).to(device)
    #         img_feature /= img_feature.norm(dim=-1, keepdim=True)
    #         img_feature = img_feature.unsqueeze(0)  # (1, D)

    #         # Compute similarity to all prompts
    #         similarity = (100.0 * img_feature @ text_features.T)  # (1, P+N)
    #         similarity_positive = similarity[:, :num_positive]
    #         similarity_positive = similarity_positive.mean(dim=1, keepdim=True)  # (1, 1)

    #         similarity_negative = similarity[:, num_positive:]
    #         similarity_negative = similarity_negative.mean(dim=1, keepdim=True)  # (1
    #         similarity = torch.cat([similarity_negative, similarity_positive], dim=1)  # (1, 2)

    #         similarities = torch.argmax(similarity, dim=1).unsqueeze(1)  # (1, 1)

    #         all_preds.append(similarities.item())
    #         all_probs.append(torch.sigmoid(similarity_positive - similarity_negative).item())

    all_preds_np = torch.tensor(all_preds).numpy()
    all_labels_np = torch.tensor(all_labels).numpy()

    f1, lower, upper, _ = evaluate_predictions(all_preds, all_labels)
    rec = recall_score(all_labels_np, all_preds_np)
    prec = precision_score(all_labels_np, all_preds_np)
    acc = accuracy_score(all_labels_np, all_preds_np)
    auc = roc_auc_score(all_labels_np, all_probs)

    print("\n--- Contrastive Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels_np, all_preds_np))
    print(
        classification_report(
            all_labels_np, all_preds_np, target_names=["No HCC", "HCC"]
        )
    )

    torch.cuda.empty_cache()
    return f1, lower, upper, rec, prec, acc, all_preds_np, all_labels_np


def evaluate_coop_model(model, val_loader, device):
    """
    Evaluation loop for trained CoOp soft prompts.

    CoOp training uses the learned class prompt order directly as CE logits:
    index 0 = negative, index 1 = HCC. Keep evaluation in the same order so
    predicted labels match the binarized dataset labels.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating CoOp performance"):
            img_data = batch["image"].to(device)
            labels = batch["labels"]
            all_labels.extend(labels.tolist())

            image_features, _, _, text_features = model(img_data, None)

            similarity = 100.0 * image_features @ text_features.T
            sim_negative = similarity[:, 0:1]
            sim_positive = similarity[:, 1:2]

            all_preds.extend(similarity.argmax(dim=1).cpu().tolist())
            all_probs.extend(
                torch.sigmoid(sim_positive - sim_negative).squeeze(1).cpu().tolist()
            )

    all_preds_np = torch.tensor(all_preds).numpy()
    all_labels_np = torch.tensor(all_labels).numpy()

    f1, lower, upper, _ = evaluate_predictions(all_preds, all_labels)
    rec = recall_score(all_labels_np, all_preds_np)
    prec = precision_score(all_labels_np, all_preds_np)
    acc = accuracy_score(all_labels_np, all_preds_np)
    auc = roc_auc_score(all_labels_np, all_probs)

    print("\n--- CoOp Soft Prompt Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels_np, all_preds_np))

    torch.cuda.empty_cache()
    return f1, lower, upper, rec, prec, acc, all_preds_np, all_labels_np


def evaluate_cocoop_model(model, val_loader, device):
    """
    Evaluation loop for image-conditioned CoCoOp soft prompts.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating CoCoOp performance"):
            img_data = batch["image"].to(device)
            labels = batch["labels"]
            all_labels.extend(labels.tolist())

            image_features, _, _, text_features = model(img_data, None)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = 100.0 * torch.einsum(
                "bd,bcd->bc", image_features, text_features
            )

            sim_negative = similarity[:, 0:1]
            sim_positive = similarity[:, 1:2]

            all_preds.extend(similarity.argmax(dim=1).cpu().tolist())
            all_probs.extend(
                torch.sigmoid(sim_positive - sim_negative).squeeze(1).cpu().tolist()
            )

    all_preds_np = torch.tensor(all_preds).numpy()
    all_labels_np = torch.tensor(all_labels).numpy()

    f1, lower, upper, _ = evaluate_predictions(all_preds, all_labels)
    rec = recall_score(all_labels_np, all_preds_np)
    prec = precision_score(all_labels_np, all_preds_np)
    acc = accuracy_score(all_labels_np, all_preds_np)
    auc = roc_auc_score(all_labels_np, all_probs)

    print("\n--- CoCoOp Soft Prompt Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels_np, all_preds_np))

    torch.cuda.empty_cache()
    return f1, lower, upper, rec, prec, acc, all_preds_np, all_labels_np


def classification_evaluation(model, val_loader, device):
    """
    Evaluate the dedicated classification head (sigmoid threshold at 0.5).

    Args:
        model: Merlin model wrapper.
        val_loader: DataLoader yielding batches with 'image' and 'labels'.
        device: Torch device string.

    Returns:
        (accuracy, recall, precision, f1)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating classification head"):
            img_data = batch["image"].to(device)
            all_labels.extend(batch["labels"].cpu().numpy())
            hcc_embedding = model.model.encode_image(img_data)[2]
            preds = (torch.sigmoid(hcc_embedding).cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds.flatten())

    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n--- HCC Classification Head Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return acc, rec, prec, f1


def evaluate_recall_at_1(
    image_features, text_features, pool_size=64, allow_partial_pool=False
):
    """
    Computes Recall@1 in both directions (image→text, text→image)
    over non-overlapping pools of given size.

    Args:
        image_features (Tensor): [N, D]
        text_features (Tensor): [N, D]
        pool_size (int): size of each evaluation pool
        allow_partial_pool (bool): whether to evaluate final pool < pool_size

    Returns:
        recall_i2t: float
        recall_t2i: float
    """
    num_samples = image_features.shape[0]
    assert image_features.shape[0] == text_features.shape[0], "Embeddings mismatch."

    image_to_text_hits = 0
    text_to_image_hits = 0
    total = 0

    for start in range(0, num_samples, pool_size):
        end = min(start + pool_size, num_samples)
        curr_size = end - start

        # Skip incomplete pool unless allowed
        if curr_size < pool_size and not allow_partial_pool:
            continue

        img_pool = image_features[start:end]
        txt_pool = text_features[start:end]

        img_pool = F.normalize(img_pool, dim=-1)
        txt_pool = F.normalize(txt_pool, dim=-1)

        sim_matrix = 100 * img_pool @ txt_pool.T  # [curr_size, curr_size]

        # Image → Text
        top_text_indices = sim_matrix.argmax(dim=1)
        image_to_text_hits += (top_text_indices == torch.arange(curr_size)).sum().item()

        # Text → Image
        top_image_indices = sim_matrix.argmax(dim=0)
        text_to_image_hits += (
            (top_image_indices == torch.arange(curr_size)).sum().item()
        )

        total += curr_size

    recall_i2t = image_to_text_hits / total
    recall_t2i = text_to_image_hits / total

    return recall_i2t, recall_t2i
