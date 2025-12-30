import os
from merlin.data import DataLoader, AugmentedDataLoader

def process_data(df, image_root, negative_root):
    datalist = []

    for _, row in df.iterrows():
        # Skip rows without venous_phase data
        if row.get("venous_phase", 0) == 0:
            continue

        imagedir = row["imagedir"]
        full_text = str(row["Content"])
        anatomy_text = str(row["Anatomies"])
        label = row["patho"]  # Still keeping this for output labeling

        found = False

        # Try 2-level directory under negative_root
        patient_dir = os.path.join(negative_root, imagedir)
        if os.path.isdir(patient_dir):
            for subfolder in os.listdir(patient_dir):
                subfolder_path = os.path.join(patient_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".nii.gz") and "venous" in filename.lower():
                        image_path = os.path.join(subfolder_path, filename)
                        datalist.append(
                            {
                                "image": image_path,
                                "full_text": full_text,
                                "anatomy_text": anatomy_text,
                                # "findings": findings,
                                "labels": label,
                            }
                        )
                        found = True
                        break
                if found:
                    break

        # If not found, try 1-level directory under image_root
        if not found:
            folder_path = os.path.join(image_root, imagedir)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".nii.gz") and "venous" in filename.lower():
                        image_path = os.path.join(folder_path, filename)
                        datalist.append(
                            {
                                "image": image_path,
                                "full_text": full_text,
                                "anatomy_text": anatomy_text,
                                "labels": label,
                            }
                        )
                        found = True
                        break

    return datalist

def get_dataloaders(train_datalist, val_datalist, batch_size, cache_dir=None):
    train_loader = DataLoader(
        train_datalist,
        batchsize=batch_size,
        cache_dir=cache_dir, # type: ignore
        shuffle=True,
    )

    val_loader = DataLoader(
        val_datalist,
        batchsize=batch_size,
        cache_dir=cache_dir, # type: ignore
        shuffle=False,
    )
    
    return train_loader, val_loader