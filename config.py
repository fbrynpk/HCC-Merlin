import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class DiseasePrompts:
    disease_name: str
    positive_prompts: List[str]
    negative_prompts: List[str]


HCC_PROMPTS = DiseasePrompts(
    disease_name="hcc",
    positive_prompts=[
        "Liver mass consistent with hepatocellular carcinoma",
        "HCC detected with typical arterial enhancement pattern",
        "Imaging shows signs of hepatocellular carcinoma",
    ],
    negative_prompts=[
        "No hepatocellular carcinoma found in liver",
        "No liver mass consistent with HCC detected",
        "Liver appears clear of hepatocellular carcinoma",
    ],
)

FRACTURE_PROMPTS = DiseasePrompts(
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Merlin VLM with HCC and Negative CT data"
    )

    # Paths
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default="path to dataset CSV",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="path to positive image root directory",
    )
    parser.add_argument(
        "--negative_root",
        type=str,
        default="path to negative image root directory",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="path to save trained model checkpoints",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=18)
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--phase",
        type=str,
        default="venous",
        choices=["venous", "all"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--tuning_mode",
        type=str,
        default="full",
        choices=["full", "lora"],
    )
    parser.add_argument(
        "--use_coop",
        action="store_true",
        help="Whether to use Context Optimization learning",
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    return parser.parse_args()
