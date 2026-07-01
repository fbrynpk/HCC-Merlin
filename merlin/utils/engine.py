import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

import wandb


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    # classification_loss_fn,
    epoch,
    device,
    global_step,
    scheduler,
    args,
    accumulation_steps=8,
):
    """
    Run one full training epoch with gradient accumulation and mixed precision.

    Alternates between 'full_text' and 'anatomy_text' keys each step to expose
    the model to both text modalities during contrastive training.

    Args:
        model: Merlin model wrapper.
        train_loader: DataLoader for the training split.
        optimizer: AdamW optimizer.
        scaler: GradScaler for mixed-precision training.
        classification_loss_fn: BCEWithLogitsLoss (or similar).
        epoch: Current epoch index (0-based).
        device: Torch device string.
        global_step: Running count of optimizer steps (for logging).
        scheduler: LR scheduler (stepped every batch, not every epoch).
        args: Parsed argument namespace.
        accumulation_steps: Number of batches to accumulate before stepping.

    Returns:
        (avg_loss, global_step, avg_cls_loss, avg_ctr_loss,
         image_features, text_features, avg_cosine_similarity)
    """
    model.train()
    total_loss = ctr_loss_sum = cls_loss_sum = 0.0
    image_embeddings, text_embeddings = [], []

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.epochs}")

    for batch in pbar:
        image = batch["image"].to(device)
        text_key = "full_text" if global_step % 2 == 0 else "anatomy_text"
        text = [t.lower() for t in batch[text_key]]

        # labels = batch["labels"].to(device).view(-1, 1).float()

        scheduler.step()
        temperature = model.model.temperature

        with autocast(dtype=torch.float16):
            img_emb, _, hcc_emb, txt_emb = model(image, text)

            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

            image_embeddings.append(img_emb.detach().cpu())
            text_embeddings.append(txt_emb.detach().cpu())

            img_logits = (img_emb @ txt_emb.T) / temperature
            n = len(img_emb)
            targets = torch.arange(n, device=device)

            ctr_loss = (
                nn.CrossEntropyLoss()(img_logits, targets)
                + nn.CrossEntropyLoss()(img_logits.T, targets)
            ) / 2
            # cls_loss = classification_loss_fn(hcc_emb, labels)
            loss = ctr_loss

        scaler.scale(loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_val = loss.detach().item()
        total_loss += loss_val
        ctr_loss_sum += ctr_loss.detach().item()
        # cls_loss_sum += cls_loss.detach().item()

        if args.use_wandb:
            wandb.log({
                "train_batch_loss": loss_val,
                "train_contrastive_loss": ctr_loss.detach().item(),
                # "train_classification_loss": cls_loss.detach().item(),
                "train_learning_rate": optimizer.param_groups[0]["lr"],
                "temperature": temperature.item(),
            }, step=global_step)

        global_step += 1

    img_feats = torch.cat(image_embeddings, dim=0)
    txt_feats = torch.cat(text_embeddings, dim=0)
    avg_cos = (img_feats * txt_feats).sum(dim=1).mean().item()

    torch.cuda.empty_cache()
    n_batches = len(train_loader)
    return (
        total_loss / n_batches,
        global_step,
        # cls_loss_sum / n_batches,
        ctr_loss_sum / n_batches,
        img_feats,
        txt_feats,
        avg_cos,
    )


def validate_one_epoch(
    model,
    val_loader,
    # classification_loss_fn,
    epoch,
    device,
    global_step,
    args,
):
    """
    Run one full validation epoch (no gradient updates).

    Args:
        model: Merlin model wrapper.
        val_loader: DataLoader for the validation split.
        classification_loss_fn: BCEWithLogitsLoss (or similar).
        epoch: Current epoch index (0-based).
        device: Torch device string.
        global_step: Running count of steps (for logging).
        args: Parsed argument namespace.

    Returns:
        (avg_loss, global_step, avg_cls_loss, avg_ctr_loss,
         image_features, text_features, avg_cosine_similarity)
    """
    model.eval()
    total_loss = ctr_loss_sum = cls_loss_sum = 0.0
    image_embeddings, text_embeddings = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{args.epochs}"):
            image = batch["image"].to(device)
            text = [t.lower() for t in batch["full_text"]]

            temperature = model.model.temperature

            with torch.amp.autocast_mode.autocast(device_type=device, dtype=torch.float16):
                img_emb, _, hcc_emb, txt_emb = model(image, text)

                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

                image_embeddings.append(img_emb.detach().cpu())
                text_embeddings.append(txt_emb.detach().cpu())

                img_logits = (img_emb @ txt_emb.T) / temperature
                n = len(img_emb)
                targets = torch.arange(n, device=device)

                ctr_loss = (
                    nn.CrossEntropyLoss()(img_logits, targets)
                    + nn.CrossEntropyLoss()(img_logits.T, targets)
                ) / 2
                # cls_loss = classification_loss_fn(hcc_emb, labels)
                loss = ctr_loss

            loss_val = loss.detach().item()
            total_loss += loss_val
            ctr_loss_sum += ctr_loss.detach().item()
            # cls_loss_sum += cls_loss.detach().item()

            if args.use_wandb:
                wandb.log({
                    "val_batch_loss": loss_val,
                    "val_contrastive_loss": ctr_loss.detach().item(),
                    # "val_classification_loss": cls_loss.detach().item(),
                    "val_temperature": temperature.item(),
                }, step=global_step)

            global_step += 1

    img_feats = torch.cat(image_embeddings, dim=0)
    txt_feats = torch.cat(text_embeddings, dim=0)
    avg_cos = (img_feats * txt_feats).sum(dim=1).mean().item()

    torch.cuda.empty_cache()
    n_batches = len(val_loader)
    return (
        total_loss / n_batches,
        global_step,
        # cls_loss_sum / n_batches,
        ctr_loss_sum / n_batches,
        img_feats,
        txt_feats,
        avg_cos,
    )