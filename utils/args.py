import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Merlin VLM with HCC and Negative CT data"
    )

    # Paths
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default="dataset.csv",
        help="Path to the dataset CSV",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="positive_folder",
        help="Root directory for HCC images",
    )
    parser.add_argument(
        "--negative_root",
        type=str,
        default="negative_folder",
        help="Root directory for Negative samples",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="checkpoint.pth",
        help="Where to save best model",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=18)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for softmax",
    )
    parser.add_argument(
        "--tuning_mode",
        type=str,
        default="full",
        choices=["full", "lora"],
        help="Fine-tuning mode",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation after training"
    )

    # WandB settings
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights and Biases for logging"
    )

    return parser.parse_args()
