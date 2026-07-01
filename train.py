import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

import wandb
from config import FRACTURE_PROMPTS, HCC_PROMPTS, parse_args
from evaluation.engine import (
    classification_evaluation,
    evaluate_model,
    evaluate_coop_model,
    evaluate_recall_at_1,
)
from merlin import Merlin
from merlin.data.dataloaders import DataLoader, VerseDataLoader, InterpolateDataLoader
from merlin.data.preprocessing import (
    binarize_labels,
    build_verse_datalist,
    process_data,
    process_data_venous,
)
from merlin.models.lora_utils import (
    apply_text_lora,
    apply_image_lora,
    freeze_module,
    conv_merge_and_unload,
    unfreeze_module,
)
from merlin.utils.engine import train_one_epoch, validate_one_epoch
from merlin.utils.optimizer import (
    build_optimizer_and_scheduler,
    build_param_groups_image_text_lldr,
)
from utils import args

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Model Building

def build_model(args, device):
    model = Merlin().to(device)
    
    if args.tuning_mode == "lora":
        freeze_module(model)

        model.model.encode_text.text_encoder = apply_text_lora(
            model.model.encode_text.text_encoder, r=16, lora_alpha=32, lora_dropout=0.2
        )
        
        model.model.encode_image.i3_resnet = apply_image_lora(
            model.model.encode_image.i3_resnet, r=2, lora_alpha=2,
        )

        unfreeze_module(model.model.encode_text.linear_layer)
        unfreeze_module(model.model.encode_image.i3_resnet.contrastive_head)

    model.model.temperature = torch.nn.Parameter(
        torch.tensor(args.temperature, device=device)
    )
    _print_param_counts(model)
    return model


def _print_param_counts(model):
    img_params = sum(
        p.numel() for p in model.model.encode_image.parameters() if p.requires_grad
    )
    txt_params = sum(
        p.numel() for p in model.model.encode_text.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable image encoder params: {img_params:,}")
    print(f"Trainable text encoder params:  {txt_params:,}")
    print(f"Total trainable params:         {total:,}")

# Data Building

def build_dataloaders(args):
    df = pd.read_csv(args.dataset_csv)
    
    # Fill NaN with 'negative' for pathology and empty string for text fields in negative samples
    df["patho"] = df["patho"].fillna("negative")
    
    # Filter to only HCC and negative samples for training/validation
    df = df[df["patho"].isin(["HCC", "negative"])].reset_index(drop=True)

    if args.phase == "venous":
        train_datalist = process_data_venous(
            df[df["split"] == "train"], args.image_root, args.negative_root
        )
        val_datalist = process_data_venous(
            df[df["split"] == "val"].sample(frac=1, random_state=42),
            args.image_root,
            args.negative_root,
        )
    else:
        train_datalist = process_data(
            df[df["split"] == "train"], args.image_root, args.negative_root
        )
        val_datalist = process_data(
            df[df["split"] == "val"].sample(frac=1, random_state=42),
            args.image_root,
            args.negative_root,
        )

    binarize_labels(train_datalist)
    binarize_labels(val_datalist)

    if args.phase == "venous":
        cache_dir = "path/to/venous_cache"
    else:
        cache_dir = None

    # Vertebral fracture loader (evaluation only)
    fracture_df = pd.read_excel("path/to/versedataset")
    fracture_df = fracture_df[fracture_df["dataset"] == "test secret"]
    verse_datalist = build_verse_datalist(
        fracture_df,
        fracture_dir="/path/to/verse_images",
    )
    
    train_loader = InterpolateDataLoader(
        train_datalist,
        batchsize=args.batch_size,
        cache_dir=cache_dir,
        shuffle=True,
        num_workers=0,
    )
    val_loader = InterpolateDataLoader(
        val_datalist,
        batchsize=args.batch_size,
        cache_dir=cache_dir,
        shuffle=False,
        num_workers=0,
    )
    verse_loader = VerseDataLoader(
        datalist=verse_datalist,
        cache_dir=None,
        batchsize=1,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, verse_loader



### Training

def train(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_wandb:
        wandb.init(
            project="merlin_vlm",
            entity="fbrynpk",
            group="Continual Learning",
            name=os.path.splitext(os.path.basename(args.model_save_path))[0],
            tags=["continual_learning", "i3_resnet", "clinical_longformer"],
            config=vars(args),
        )

    model = build_model(args, device)
    train_loader, val_loader, verse_loader = build_dataloaders(args)

    # Build LLDR param groups
    image_layer_names = [n for n, _ in model.model.encode_image.named_parameters()][
        ::-1
    ]
    text_layer_names = [n for n, _ in model.model.encode_text.named_parameters()][::-1]

    param_groups = build_param_groups_image_text_lldr(
        model=model,
        image_layer_names=image_layer_names,
        text_layer_names=text_layer_names,
        base_lr_img=args.learning_rate,
        base_lr_txt=args.learning_rate,
        lr_mult_img=0.7,
        lr_mult_txt=0.7,
        weight_decay=0.01,
    )

    total_steps = args.epochs * len(train_loader)
    
    # 1% warmup
    # warmup_steps = int(0.01 * total_steps)
    
    optimizer, scheduler, scaler = build_optimizer_and_scheduler(
        model, param_groups, args, total_steps, warmup_steps=0
    )
    # classification_loss_fn = torch.nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_val_loss = float("inf")
    
    best_loss_path = args.model_save_path.replace(".pth", "_best_loss.pth")
    best_f1_path = args.model_save_path.replace(".pth", "_best_f1.pth")

    patience_counter = 0
    global_step = 0

    for epoch in range(args.epochs):
        (
            train_loss,
            global_step,
            # cls_loss,
            ctr_loss,
            train_img_feats,
            train_txt_feats,
            train_cos,
        ) = train_one_epoch(
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
        )

        (
            val_loss,
            global_step,
            # val_cls_loss,
            val_ctr_loss,
            val_img_feats,
            val_txt_feats,
            val_cos,
        ) = validate_one_epoch(
            model,
            val_loader,
            # classification_loss_fn,
            epoch,
            device,
            global_step,
            args,
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        f1_score = 0.0
        if args.evaluate:
            # acc_cls, rec_cls, prec_cls, f1_cls = classification_evaluation(
                # model, val_loader, device
            # )

            f1_score, lower, upper, recall, precision, accuracy, _, _ = evaluate_model(
                model, val_loader, device, HCC_PROMPTS
            )
            f1_frac, lo_frac, hi_frac, rec_frac, prec_frac, acc_frac, _, _ = (
                evaluate_model(model, verse_loader, device, FRACTURE_PROMPTS)
            )
            (train_recall_i2t, train_recall_t2i) = evaluate_recall_at_1(
                torch.tensor(train_img_feats),
                torch.tensor(train_txt_feats),
                pool_size=64,
                allow_partial_pool=True,
            )

            (val_recall_i2t, val_recall_t2i) = evaluate_recall_at_1(
                torch.tensor(val_img_feats),
                torch.tensor(val_txt_feats),
                pool_size=64,
                allow_partial_pool=True,
            )
            if args.use_wandb:
                wandb.log(
                    {
                        "train/epoch_loss": train_loss,
                        "train/epoch_contrastive_loss": ctr_loss,
                        # "train/epoch_classification_loss": cls_loss,
                        "train/epoch_cosine_similarity": train_cos,
                        "val/epoch_loss": val_loss,
                        "val/epoch_contrastive_loss": val_ctr_loss,
                        # "val/epoch_classification_loss": val_cls_loss,
                        "val/epoch_cosine_similarity": val_cos,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "evaluate/F1-Score": f1_score,
                        "evaluate/F1-Lower": lower,
                        "evaluate/F1-Upper": upper,
                        "evaluate/Recall": recall,
                        "evaluate/Precision": precision,
                        "evaluate/Accuracy": accuracy,
                        "evaluate_fracture/F1-Score": f1_frac,
                        "evaluate_fracture/F1-Lower": lo_frac,
                        "evaluate_fracture/F1-Upper": hi_frac,
                        "evaluate_fracture/Recall": rec_frac,
                        "evaluate_fracture/Precision": prec_frac,
                        "evaluate_fracture/Accuracy": acc_frac,
                        "val/Recall@1-ImageToText": val_recall_i2t,
                        "val/Recall@1-TextToImage": val_recall_t2i,
                        "train/Recall@1-ImageToText": train_recall_i2t,
                        "train/Recall@1-TextToImage": train_recall_t2i,
                    }
                )

        # Checkpoint on best F1
        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model.state_dict(), best_f1_path)
            print(f"Saved best model (F1={f1_score:.4f})")
            patience_counter = 0
        else:
            print(f"No improvement (best F1={best_f1:.4f})")
            patience_counter += 1

        # Early stopping
        if patience_counter >= 5:
            print("Early stopping triggered.")
            break

    # Merge LoRA adapters and save clean model
    if args.tuning_mode == "lora":
        model.load_state_dict(torch.load(best_loss_path))
        model.model.encode_text.text_encoder = (
            model.model.encode_text.text_encoder.merge_and_unload()
        )
        model.model.encode_image.i3_resnet = conv_merge_and_unload(
            model.model.encode_image.i3_resnet
        )
        merged_path = best_loss_path.replace(".pth", "_merged.pth")
        torch.save(model.state_dict(), merged_path)
        print(f"Merged best loss model saved to {merged_path}")
        
        # Merge best F1 model as well (if different)
        model = build_model(args, device)
        model.load_state_dict(torch.load(best_f1_path))
        model.model.encode_text.text_encoder = (
            model.model.encode_text.text_encoder.merge_and_unload()
        )
        model.model.encode_image.i3_resnet = conv_merge_and_unload(
            model.model.encode_image.i3_resnet
        )
        merged_f1_path = best_f1_path.replace(".pth", "_merged.pth")
        torch.save(model.state_dict(), merged_f1_path)
        print(f"Merged best F1 model saved to {merged_f1_path}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train(parse_args())
