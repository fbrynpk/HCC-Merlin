import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from metrics import evaluate_predictions

# Evaluation on validation set
def evaluate_model(model, val_loader, device, prompts):
    model.eval()

    image_features_all = []
    all_preds = []
    all_labels = []
    num_positive_prompts = len(prompts.positive_prompts)

    with torch.no_grad():
        text_features = model.model.encode_text(
            prompts.positive_prompts + prompts.negative_prompts
        )

        for i, batch in tqdm(
            enumerate(val_loader), total=len(val_loader), desc="Evaluating"
        ):
            img_data = batch["image"].to(device)
            batch_labels = batch["labels"].to(device)
            all_labels.extend(batch_labels)
            image_features = model.model.encode_image(img_data)[0]

            for img_feature in image_features:
                image_features_all.append(img_feature.cpu().numpy())

        text_features /= text_features.norm(dim=-1, keepdim=True)

        for img_feature in image_features_all:
            img_feature = torch.tensor(img_feature).to(device)
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            img_feature = img_feature.unsqueeze(0)

            # Compute the similarity between the image and text embeddings
            similarity = 100.0 * img_feature @ text_features.T
            similarity_positive = similarity[:, :num_positive_prompts]
            similarity_positive = similarity_positive.mean(dim=1, keepdim=True)

            similarity_negative = similarity[:, num_positive_prompts:]
            similarity_negative = similarity_negative.mean(dim=1, keepdim=True)
            similarity = torch.cat([similarity_negative, similarity_positive], dim=1)

            similarities = torch.argmax(similarity, dim=1).unsqueeze(1)

            all_preds.append(similarities.item())
        torch.cuda.empty_cache()
        # print("Split Type: ", split_type)
        f1_score, lower, upper, bootstrapped_f1_scores = evaluate_predictions(
            all_preds, all_labels
        )

    # Compute Recall and Precision
    recall = recall_score(
        torch.tensor(all_labels).cpu().numpy(), torch.tensor(all_preds).cpu().numpy()
    )
    precision = precision_score(
        torch.tensor(all_labels).cpu().numpy(), torch.tensor(all_preds).cpu().numpy()
    )

    # Compute Accuracy
    accuracy = accuracy_score(
        torch.tensor(all_labels).cpu().numpy(), torch.tensor(all_preds).cpu().numpy()
    )

    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(
        torch.tensor(all_labels).cpu().numpy(), torch.tensor(all_preds).cpu().numpy()
    )
    print("Confusion Matrix:")
    print(conf_matrix)
    # Print classification report
    print("Classification Report:")
    print(
        classification_report(
            torch.tensor(all_labels).cpu().numpy(),
            torch.tensor(all_preds).cpu().numpy(),
            target_names=["No HCC", "HCC"],
        )
    )

    return (
        f1_score,
        lower,
        upper,
        recall,
        precision,
        accuracy,
        torch.tensor(all_preds).cpu().numpy(),
        torch.tensor(all_labels).cpu().numpy(),
    )