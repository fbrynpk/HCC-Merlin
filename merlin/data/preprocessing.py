import os
import random
import re

import pandas as pd


def process_data_venous(df, image_root, negative_root):
    """Build datalist using only venous-phase NIfTI files."""
    datalist = []

    for _, row in df.iterrows():
        if row.get("venous_phase", 0) == 0:
            continue

        imagedir = row["imagedir"]
        full_text = str(row["Content"])
        anatomy_text = str(row["Anatomies"])
        # findings = str(row["Findings"])
        # impression = str(row["Impression"])
        # full_text = f"Findings: {findings}\n\nImpression: {impression}"
        label = row["patho"]

        found = False

        # Try 2-level directory under negative_root first
        patient_dir = os.path.join(negative_root, imagedir)
        if os.path.isdir(patient_dir):
            for subfolder in os.listdir(patient_dir):
                subfolder_path = os.path.join(patient_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".nii.gz") and "venous" in filename.lower():
                        datalist.append({
                            "image": os.path.join(subfolder_path, filename),
                            "full_text": full_text,
                            "anatomy_text": anatomy_text,
                            "labels": label,
                        })
                        found = True
                        break
                if found:
                    break

        # Fall back to 1-level directory under image_root
        if not found:
            folder_path = os.path.join(image_root, imagedir)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".nii.gz") and "venous" in filename.lower():
                        datalist.append({
                            "image": os.path.join(folder_path, filename),
                            "full_text": full_text,
                            "anatomy_text": anatomy_text,
                            "labels": label,
                        })
                        break

    return datalist


def process_data(df, image_root, negative_root):
    """Build datalist using all available NIfTI files (all phases)."""
    datalist = []

    for _, row in df.iterrows():
        imagedir = row["imagedir"]
        full_text = str(row["Content"])
        anatomy_text = str(row["Anatomies"])
        label = row["patho"]

        # Try negative_root (2-level directories)
        patient_dir = os.path.join(negative_root, imagedir)
        if os.path.isdir(patient_dir):
            for subfolder in os.listdir(patient_dir):
                subfolder_path = os.path.join(patient_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".nii.gz"):
                        datalist.append({
                            "image": os.path.join(subfolder_path, filename),
                            "full_text": full_text,
                            "anatomy_text": anatomy_text,
                            "labels": label,
                            "patient_id": row["imagedir"],
                        })

        # Try image_root (1-level directories)
        folder_path = os.path.join(image_root, imagedir)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".nii.gz"):
                    datalist.append({
                        "image": os.path.join(folder_path, filename),
                        "full_text": full_text,
                        "anatomy_text": anatomy_text,
                        "labels": label,
                        "patient_id": row["imagedir"],
                    })

    return datalist


def build_verse_datalist(fracture_df, fracture_dir):
    """
    Match vertebra fracture rows to their NIfTI files on disk.

    Args:
        fracture_df: DataFrame with columns subject_ID, verse_ID, fracture.
        fracture_dir: Directory containing the cropped NIfTI files.

    Returns:
        List of dicts with keys 'image' and 'labels'.
    """
    available_files = set(os.listdir(fracture_dir))
    datalist = []

    for _, row in fracture_df.iterrows():
        subject_id = row["subject_ID"]
        verse_id = row["verse_ID"]
        fname = f"subj_{subject_id:03d}_verse_{verse_id:03d}_ct_crop.nii.gz"

        if fname not in available_files:
            raise FileNotFoundError(
                f"No matching file for subject_ID={subject_id}, verse_ID={verse_id}"
            )

        datalist.append({
            "image": os.path.join(fracture_dir, fname),
            "labels": row["fracture"],
        })

    return datalist


def binarize_labels(datalist):
    """Convert 'HCC'/'negative' string labels to 1/0 integers in-place."""
    for data in datalist:
        data["labels"] = 1 if data["labels"] == "HCC" else 0
    return datalist


def stratified_sample(datalist, ratio):
    """Sample a fixed ratio of positives and negatives, preserving class balance."""
    pos = [x for x in datalist if x["labels"] == 1]
    neg = [x for x in datalist if x["labels"] == 0]

    pos_sample = random.sample(pos, max(1, int(len(pos) * ratio)))
    neg_sample = random.sample(neg, max(1, int(len(neg) * ratio)))

    return pos_sample + neg_sample