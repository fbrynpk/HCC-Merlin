import argparse
import os
import random
import re
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from dpipe.layers.conv import PreActivationND
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm.auto import tqdm

import lora_layers as lora
from merlin import Merlin
from merlin.data.dataloaders import DataLoader, VerseDataLoader
from merlin.models.i3res import Bottleneck3d
from utils import args

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# SET SEED
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    epoch,
    device,
    global_step,
    scheduler,
    args,
):
    model.train()
    epoch_loss = 0
    contrastive_loss_epoch = 0
    accumulation_steps = 8
    image_embeddings = []
    text_embeddings = []
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.epochs}")

    for batch in pbar:
        image = batch["image"].to(device)
        text_key = "full_text" if global_step % 2 == 0 else "anatomy_text"
        text = [t.lower() for t in batch[text_key]]

        scheduler.step()
        temperature = model.model.temperature
        # logit_scale = model.model.logit_scale.exp()
        with autocast(dtype=torch.float16):
            image_embedding, phenotypes, text_embedding = model(image, text)
            
            # Normalized
            image_embedding = image_embedding / image_embedding.norm(
                dim=-1, keepdim=True
            )
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            image_embeddings.append(image_embedding.detach().cpu())
            text_embeddings.append(text_embedding.detach().cpu())

            # Cosine similarity logits
            image_logits =  (image_embedding @ text_embedding.T) / temperature
            text_logits = image_logits.T
            
            loss_i2t = nn.CrossEntropyLoss()(
                image_logits, torch.arange(len(image_embedding), device=device)
            )

            loss_t2i = nn.CrossEntropyLoss()(
                text_logits, torch.arange(len(text_embedding), device=device)
            )
            
            contrastive_loss = (loss_i2t + loss_t2i) / 2
        
        loss_value = contrastive_loss.detach().item()
        scaler.scale(contrastive_loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss_value
        contrastive_loss_epoch += contrastive_loss.detach().item()

        if args.use_wandb:
            wandb.log(
                {"train_contrastive_loss": contrastive_loss.detach().item()}, step=global_step
            )
            wandb.log(
                {"train_learning_rate": optimizer.param_groups[0]["lr"]},
                step=global_step,
            )
            wandb.log({"temperature": temperature.item()}, step=global_step)

        global_step += 1
        del image, text, image_embedding, text_embedding, contrastive_loss, phenotypes, image_logits, text_logits, loss_i2t, loss_t2i
    
    image_features = torch.cat(image_embeddings, dim=0)
    text_features = torch.cat(text_embeddings, dim=0)

    cos_sim = torch.sum(image_features * text_features, dim=1)
    avg_cosine_similarity = cos_sim.mean().item()
    
    del image_embeddings, text_embeddings
    torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    return (
        (epoch_loss / len(train_loader)),
        global_step,
        contrastive_loss_epoch / len(train_loader),
        avg_cosine_similarity,
    )


def validate_one_epoch(
    model,
    val_loader,
    epoch,
    device,
    global_step,
    args,
):
    model.eval()
    val_epoch_loss = 0
    val_contrastive_loss_epoch = 0
    image_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc=f"Validating Epoch {epoch + 1}/{args.epochs}"
        ):
            image = batch["image"].to(device)
            # phase = batch["phase"]
            text = [t.lower() for t in batch["full_text"]]

            temperature = model.model.temperature
            # logit_scale = model.model.logit_scale.exp()
            with torch.amp.autocast_mode.autocast(
                device_type=device, dtype=torch.float16
            ):
                image_embedding, phenotypes, text_embedding = model(image, text)

                # Normalized
                image_embedding = image_embedding / image_embedding.norm(
                    dim=-1, keepdim=True
                )
                text_embedding = text_embedding / text_embedding.norm(
                    dim=-1, keepdim=True
                )

                image_embeddings.append(image_embedding.detach().cpu())
                text_embeddings.append(text_embedding.detach().cpu())

                # Cosine similarity logits
                image_logits =  (image_embedding @ text_embedding.T) / temperature
                text_logits = image_logits.T
                
                loss_i2t = nn.CrossEntropyLoss()(
                    image_logits, torch.arange(len(image_embedding), device=device)
                )

                loss_t2i = nn.CrossEntropyLoss()(
                    text_logits, torch.arange(len(text_embedding), device=device)
                )

                val_contrastive_loss = (loss_i2t + loss_t2i) / 2

            val_loss_value = val_contrastive_loss.detach().item()
            val_epoch_loss += val_loss_value
            val_contrastive_loss_epoch += val_contrastive_loss.detach().item()

            if args.use_wandb:
                wandb.log(
                    {"val_contrastive_loss": val_contrastive_loss.detach().item()},
                    step=global_step,
                )
                wandb.log({"val_temperature": temperature.item()}, step=global_step)

            global_step += 1
            del image, text, image_embedding, text_embedding, val_contrastive_loss, phenotypes, image_logits, text_logits, loss_i2t, loss_t2i
        torch.cuda.empty_cache()
        image_features = torch.cat(image_embeddings, dim=0)
        text_features = torch.cat(text_embeddings, dim=0)

        cos_sim = torch.sum(image_features * text_features, dim=1)
        avg_cosine_similarity = cos_sim.mean().item()
        
        del image_embeddings, text_embeddings
        torch.cuda.empty_cache()

    return (
        (val_epoch_loss / len(val_loader)),
        global_step,
        (val_contrastive_loss_epoch / len(val_loader)),
        avg_cosine_similarity,
    )


def lr_lambda_warmup(current_step: int, warmup_steps: int = 1000):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

def apply_convlora(model, desired_submodules=None, r=2, alpha=2, dropout=0.0, merge_weights=True):
    """
    Recursively replace Conv3d in selected submodules with ConvLoRA(conv_module=nn.Conv3d)
    while preserving pretrained weights.
    """

    def wrap_conv3d(parent, name, old_conv):
        """Replace an nn.Conv3d with ConvLoRA(conv_module=nn.Conv3d) safely."""

        # Create new ConvLoRA wrapper
        new_conv = lora.Conv3d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size[0],
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge_weights=merge_weights,
        )

        # Copy pretrained weights & bias into new conv module
        # The ConvLoRA stores the real conv as new_conv.conv
        new_conv.conv.weight.data.copy_(old_conv.weight.data)

        if old_conv.bias is not None:
            new_conv.conv.bias.data.copy_(old_conv.bias.data)

        # Replace in parent module
        setattr(parent, name, new_conv)

    def recursive_replace(module, module_name="", restrict_to_top=True):
        for child_name, child in list(module.named_children()):

            # Restrict only at top level
            if restrict_to_top and desired_submodules is not None:
                if child_name not in desired_submodules:
                    continue

            # (1) Direct Conv3d → wrap it
            if isinstance(child, nn.Conv3d):
                wrap_conv3d(module, child_name, child)
                continue

            # (2) PreActivationND.layer contains Conv3d
            if isinstance(child, PreActivationND) and isinstance(child.layer, nn.Conv3d):
                wrap_conv3d(child, "layer", child.layer)
                continue

            # (3) Bottleneck3d → recurse into its conv_path + other children
            if isinstance(child, Bottleneck3d):
                recursive_replace(child, module_name=f"{module_name}.{child_name}", restrict_to_top=False)
                continue

            # (4) Generic recursion for other submodules (Sequential, custom blocks, etc.)
            recursive_replace(child, module_name=f"{module_name}.{child_name}", restrict_to_top=False)

    # Begin replacement from the model root
    recursive_replace(model, restrict_to_top=True)
    return model


def merge_i3_resnet_convlora(i3_resnet: nn.Module) -> nn.Module:
    """
    Recursively merge ConvLoRA Conv3d layers inside i3_resnet into their base nn.Conv3d
    and strip away the LoRA wrappers, returning a pure Conv3d backbone.
    """

    def _merge_in_module(module: nn.Module):
        for name, child in list(module.named_children()):
            # Case 1: this is a LoRA Conv3d wrapper
            if isinstance(child, lora.Conv3d):  # <-- your Conv3d class, not nn.Conv3d
                # Ensure LoRA weights are merged into conv.weight
                if child.r > 0 and child.merge_weights and not child.merged:
                    # This calls ConvLoRA.train(False) logic:
                    child.eval()   # triggers merge into child.conv.weight

                # Now child.conv is a plain nn.Conv3d with merged weights
                base_conv = child.conv

                # Optionally re-enable grads if you ever want to fine-tune again later
                for p in base_conv.parameters():
                    p.requires_grad = True

                # Replace the wrapper in its parent with the bare Conv3d
                setattr(module, name, base_conv)

            else:
                # Recurse into children
                _merge_in_module(child)

    _merge_in_module(i3_resnet)
    return i3_resnet


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True

def main():
    set_seed(42)
    args = args.parse_args()

    # Set up
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize wandb
    if args.use_wandb:
        wandb.init(
            project="merlin_vlm",
            entity="fbrynpk",
            group="Continual Learning",
            name=args.model_save_path.split("/")[-1].split(".")[0],
            tags=["continual_learning", "i3_resnet", "clinical_longformer"],
            config={
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "loss": "InfoNCE",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            },
        )

    # Model init
    model = Merlin().to(device)

    if args.tuning_mode == "lora":
        # Freeze model before adding LoRA
        for param in model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            target_modules=[
                "query",
                "key",
                "value",
            ],
            lora_dropout=0.2,
            bias="none",
        )

        model.model.encode_text.text_encoder = get_peft_model(
            model.model.encode_text.text_encoder, lora_config
        )
        model.model.encode_text.text_encoder.config.gradient_checkpointing = True

        # ConvLoRA
        
        model.model.encode_image.i3_resnet = apply_convlora(
            model.model.encode_image.i3_resnet,
            desired_submodules=["layer1", "layer2", "layer3", "layer4"],
            r=2,
            alpha=2,
        ).to(device)
        
        # Unfreeze last layer of text encoeder
        unfreeze_module(model.model.encode_text.linear_layer)
        
        # Unfreeze contrastive head of image encoder
        unfreeze_module(model.model.encode_image.i3_resnet.contrastive_head)
        
        # Add learnable temperature parameters
        model.model.temperature = nn.Parameter(torch.tensor(args.temperature, device=device))

        print(
            f"Trainable parameters image encoder: {sum(p.numel() for p in model.model.encode_image.parameters() if p.requires_grad)}"
        )
        print(
            f"Trainable parameters text encoder: {sum(p.numel() for p in model.model.encode_text.parameters() if p.requires_grad)}"
        )
        print(
            f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
    else:
        model.model.temperature = nn.Parameter(torch.tensor(args.temperature, device=device))

        print(
            f"Trainable parameters image encoder: {sum(p.numel() for p in model.model.encode_image.parameters() if p.requires_grad)}"
        )
        print(
            f"Trainable parameters text encoder: {sum(p.numel() for p in model.model.encode_text.parameters() if p.requires_grad)}"
        )
        print(
            f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
    
    # Data preparation
    cache_dir = os.path.join("/media/ryan/TOSHIBA1", "venous_phase_cache")
    df = pd.read_csv(args.dataset_csv)
    fracture_df = pd.read_excel("/media/ryan/T500/verse/verse_fracture.xlsx")
    
    # HCC Dataset Preprocess
    df = df[df["patho"].isin(["HCC", "negative"])].reset_index(drop=True)
    train_datalist = process_data(
        df[df["split"] == "train"], args.image_root, args.negative_root
    )
    val_datalist = process_data(
        df[df["split"] == "val"].sample(frac=1, random_state=42),
        args.image_root,
        args.negative_root,
    )

    
    # Convert patho to binary labels
    for data in train_datalist:
        data["labels"] = 1 if data["labels"] == "HCC" else 0
    for data in val_datalist:
        data["labels"] = 1 if data["labels"] == "HCC" else 0
            
    # Fracture Dataset Preprocess
    fracture_df = fracture_df[fracture_df["dataset"] == 'test secret']
    fracture_gt = fracture_df["fracture"].tolist()
    
    fracture_dir = "/media/ryan/T500/Merlin/verse_code/Verse/ct_verse_extracted"
    available_files = set(os.listdir(fracture_dir))
    # Construct path for each row
    nifti_paths = []
    for _, row in fracture_df.iterrows():
        subject_id = row["subject_ID"]
        verse_id = row["verse_ID"]
        
        candidates = [f"subj_{subject_id:03d}_verse_{verse_id:03d}_ct_crop.nii.gz"]

        matched = None
        for fname in candidates:
            if fname in available_files:
                matched = fname
                break

        if matched is None:
            raise FileNotFoundError(
                f"No matching file for subject_ID={subject_id}, verse_ID={verse_id}"
            )

        nifti_paths.append(os.path.join(fracture_dir, matched))
    # print(len(nifti_paths))

    # Check if the number of NIfTI files matches the number of ground truth labels
    assert len(nifti_paths) == len(fracture_gt), (
        "Number of NIfTI files does not match number of ground truth labels"
    )

    # Create datalist for NIfTI paths
    verse_datalist = [
        {
            "image": path,
            "labels": label,
        }
        for path, label in zip(nifti_paths, fracture_gt)
    ]

    # Load DataLoader

    train_loader = DataLoader(
        train_datalist,
        batchsize=args.batch_size,
        cache_dir=cache_dir,  
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_datalist,
        batchsize=args.batch_size,
        cache_dir=cache_dir,  
        shuffle=False,
        num_workers=0,
    )
    verse_loader = VerseDataLoader(
        verse_datalist,
        cache_dir=None, 
        batchsize=1,
        shuffle=False,
        num_workers=0,
    )

    total_steps = args.epochs * len(train_loader)
    # warmup_steps = int(0.1 * total_steps)  # Warmup for 10% of total steps
    warmup_steps = 0

    exclude = (
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "LayerNorm" in n
        or "bias" in n
        or "logit_scale" in n
        or "temperature" in n
    )
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": 0.01},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
    )

    # Warmup
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda_warmup(step, warmup_steps=warmup_steps),
    )

    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=0
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )
    scaler = GradScaler()
    best_val_loss = float("inf")
    best_f1_score = 0.0
    global_step = 0
    patience_counter = 0

    @dataclass
    class DiseasePrompts:
        disease_name: str
        positive_prompts: List[str]
        negative_prompts: List[str]

    # Define prompts for fracture detection
    hcc_prompts = DiseasePrompts(
        disease_name="hcc",
        positive_prompts=[
            "Liver mass consistent with hepatocellular carcinoma",
            "HCC detected with typical arterial enhancement pattern",
            "Imaging shows signs of hepatocellular carcinoma",
        ],
        # Define prompts for no fracture detection
        negative_prompts=[
            "No hepatocellular carcinoma found in liver",
            "No liver mass consistent with HCC detected",
            "Liver appears clear of hepatocellular carcinoma",
        ],
    )
    fracture_prompts = DiseasePrompts(
        disease_name="fracture",
        positive_prompts=[
            "compression fracture",
            "fracture identified",
            "fractures identified",
            "osteoporotic fracture",
            "vertebral fracture",
            "thoracic fracture",
            "thoracolumbar fracture",
            "wedge fractture",
            "biconcave fracture",
            "crush fracture",
        ],
        negative_prompts=[
            "no fracture",
            "no displaced fracture",
            "no acute fracture",
            "no evidence of fracture",
            "without evidence of fracture",
            "deformity and developmental abnormality",
        ],
    )

    for epoch in range(args.epochs):
        (
            train_loss,
            global_step,
            contrastive_loss,
            avg_cosine_similarity,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch,
            device,
            global_step,
            scheduler,
            args,
        )

        (
            val_loss,
            global_step,
            val_contrastive_loss,
            val_avg_cosine_similarity,
        ) = validate_one_epoch(
            model,
            val_loader,
            epoch,
            device,
            global_step,
            args,
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
        )


        if args.evaluate:
            (
                f1_score,
                lower,
                upper,
                recall,
                precision,
                accuracy,
                all_preds,
                all_labels,
            ) = evaluate_model(model, val_loader, device, hcc_prompts)
            
            # Evaluate for fracture detection
            (
                f1_score_fracture,
                lower_fracture,
                upper_fracture,
                recall_fracture,
                precision_fracture,
                accuracy_fracture,
                all_preds_fracture,
                all_labels_fracture,
            ) = evaluate_model(model, verse_loader, device, fracture_prompts)

        # wandb logging for epoch
        if args.use_wandb:
            wandb.log(
                {
                    "train/epoch_loss": train_loss,
                    "train/epoch_contrastive_loss": contrastive_loss,
                    "train/epoch_cosine_similarity": avg_cosine_similarity,
                    "val/epoch_loss": val_loss,
                    "val/epoch_contrastive_loss": val_contrastive_loss,
                    "val/epoch_cosine_similarity": val_avg_cosine_similarity,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "evaluate/F1-Score": f1_score,
                    "evaluate/F1-Lower": lower,
                    "evaluate/F1-Upper": upper,
                    "evaluate/Recall": recall,
                    "evaluate/Precision": precision,
                    "evaluate/Accuracy": accuracy,
                    "evaluate_fracture/F1-Score": f1_score_fracture,
                    "evaluate_fracture/F1-Lower": lower_fracture,
                    "evaluate_fracture/F1-Upper": upper_fracture,
                    "evaluate_fracture/Recall": recall_fracture,
                    "evaluate_fracture/Precision": precision_fracture,
                    "evaluate_fracture/Accuracy": accuracy_fracture,
                }
            )
        
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Saved best model (F1-Score={f1_score:.4f}), Val Loss={val_loss:.4f}")
            patience_counter = 0
        else:
            print(f"No improvement in F1-Score (best_F1-Score={best_f1_score:.4f})")
            patience_counter += 1
            
        if patience_counter >= 5:
            print("Early stopping triggered")
            break
        
    if args.tuning_mode == "lora":
        # Load best weights before saving
        model.load_state_dict(torch.load(args.model_save_path))
        
        # Merge LoRA weights back into the base model
        # merge_image_encoder = model.model.encode_image.i3_resnet.get_merged_model()
        # merge_image_encoder = merge_i3_resnet_convlora(model.model.encode_image.i3_resnet)
        merge_text_encoder = model.model.encode_text.text_encoder.merge_and_unload()
        
        # model.model.encode_image.i3_resnet = merge_image_encoder
        model.model.encode_text.text_encoder = merge_text_encoder
        
        # Save the merged model
        torch.save(model.state_dict(), args.model_save_path.replace(".pth", "_merged.pth"))

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
