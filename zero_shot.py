import os

import pandas as pd
import torch
import torch.nn as nn

from config import HCC_PROMPTS
from evaluation.engine import (
    classification_evaluation,
    evaluate_model,
)
from merlin import Merlin
from merlin.data import DataLoader
from merlin.data.dataloaders import DataLoader, InterpolateDataLoader
from merlin.data.preprocessing import (
    process_data_venous,
)

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Merlin model
model = Merlin()
model.eval()
model.to(device)

# Add learnable temperature parameters
model.model.temperature = nn.Parameter(torch.tensor(10.0, device=device))

model_path = "path to model checkpoint"  # Path to trained model checkpoint

# Load model state
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Excel Path
excel_path = "path to dataset CSV"
image_root = "path to positive image root directory"
negative_root = "path to negative image root directory"

# read img_dir, labels
df = pd.read_csv(excel_path)
df = df[df["patho"].isin(["HCC", "negative"])].reset_index(drop=True)

ground_truths = []
nifti_paths = []

# Create datalist for NIfTI paths
test_datalist = process_data_venous(
    df[df["split"] == "test"], image_root, negative_root
)

for data in test_datalist:
    data["labels"] = 1 if data["labels"] == "HCC" else 0

cache_dir = "path/to/venous_cache"  # Optional cache directory for preprocessed data
# Load DataLoader
dataloader = InterpolateDataLoader(
    datalist=test_datalist,
    cache_dir=cache_dir,
    batchsize=1,
    shuffle=False,
    num_workers=0,
)


def main():
    evaluate_model(model, dataloader, device, HCC_PROMPTS)
    # classification_evaluation(model, dataloader, device)


if __name__ == "__main__":
    main()
