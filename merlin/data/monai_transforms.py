import torch
import torch.nn.functional as F
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CropForegroundd,
    Resized,
    RandSpatialCropd,
)
# Z-axis interpolation to generate realistic axial slices
def interpolate_z_axis(volume, target_depth):
    volume = volume.unsqueeze(0)  # (1, C, H, W, D)
    volume = volume.permute(0, 1, 4, 2, 3)  # (1, C, D, H, W)
    interpolated = F.interpolate(
        volume,
        size=(target_depth, volume.shape[-2], volume.shape[-1]),
        mode="trilinear",
        align_corners=True,
    )
    interpolated = interpolated.permute(0, 1, 3, 4, 2)  # (1, C, H, W, D)
    return interpolated.squeeze(0)


class InterpolateZAxis:
    def __init__(self, keys=["image"], target_depth=160):
        self.keys = keys
        self.target_depth = target_depth

    def __call__(self, data):
        for key in self.keys:
            img = data[key]  # shape: (C, H, W, D)
            if not torch.is_tensor(img):
                img = torch.tensor(img)
            data[key] = interpolate_z_axis(img, self.target_depth)
        return data

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
        SpatialPadd(keys=["image"], spatial_size=[224, 224, -1]),
        InterpolateZAxis(keys=["image"], target_depth=160),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        ToTensord(keys=["image"]),
    ]
)
    
InterpolateImageTransforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear",)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, -1]),
        InterpolateZAxis(keys=["image"], target_depth=160),
        CenterSpatialCropd(
            keys=["image"], roi_size=[224, 224, 160]
        ),
        ToTensord(keys=["image"]),
    ]
)
