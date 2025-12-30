import torch
import torch.nn.functional as F
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    MapTransform
)

ImageTransforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        ToTensord(keys=["image"]),
    ]
)

VerseImageTransforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        ToTensord(keys=["image"]),
    ]
)

