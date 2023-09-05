import torch
from monai.data import NibabelReader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    SpatialCropd,
    SpatialPadd,
    ToTensord,
)


def get_preprocessing(device: torch.device) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image"], reader=NibabelReader()),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="LAS"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            SpatialCropd(
                keys=["image"],
                roi_start=[40, 12, 80],
                roi_end=[200, 236, 240],
            ),
            SpatialPadd(
                keys=["image"],
                spatial_size=[160, 224, 160],
            ),
            ToTensord(keys=["image"], device=device),
        ]
    )
