import glob
import os
from sklearn import model_selection
import nibabel as nib
import numpy as np
import torch

from monai.data import (DataLoader, CacheDataset)
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    Resized,
    NormalizeIntensityd,
    ToTensord,
    AsDiscrete,
)

from util import MinMax

RANDOM_SEED = 831
VAL_RATIO = 0.2


def get_dataset(data_type, image_size, batch_size):

    if data_type == 'decathron_spleen':
        data_dir = "./data/Task09_Spleen"

    elif data_type == 'decathron_colon':
        data_dir = "./data/Task10_Colon"

    elif data_type == 'decathron_heart':
        data_dir = "./data/Task02_Heart"

    img_path_list = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    mask_path_list = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    img_mask_dicts = [{"img" : img_name, "mask": mask_name} for img_name, mask_name in zip(img_path_list, mask_path_list)]

    train_dicts, val_dicts = model_selection.train_test_split(img_mask_dicts, test_size=VAL_RATIO, random_state=RANDOM_SEED)
    print("train, valset classify complete")

    ## sample check
    # for i in range(2):
    #     sample_img = nib.load(train_dicts[i]['img']).get_fdata()
    #     sample_mask = nib.load(train_dicts[i]['mask']).get_fdata()
    #     print("------------sample image, mask config-------------")
    #     print(f"[sample {i+1}] {os.path.basename(train_dicts[i]['img'])} {os.path.basename(train_dicts[i]['mask'])}")
    #     print(sample_img.shape, sample_img.dtype, np.min(sample_img), np.max(sample_img))
    #     print(sample_mask.shape, sample_mask.dtype, np.unique(sample_mask))
    #     print("--------------------------------------------------")
    
    transforms = Compose([LoadImaged(keys=('img','mask'), image_only=False),
                            EnsureChannelFirstd(keys=["img","mask"]),
                            Orientationd(keys=['img','mask'], axcodes='RAS'),
                            # Spacingd(keys=["image", "label"], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")), # don't have pixdim in dataset
                            Resized(keys=["img",], spatial_size=(image_size), mode='trilinear'),
                            Resized(keys=['mask',], spatial_size=(image_size), mode='nearest-exact'),
                            # NormalizeIntensityd(keys=["img",]),
                            MinMax(keys=['img',]),
                            ToTensord(keys=["img", "mask"]),
                            ])

    ## sample check
    # sample = transforms(train_dicts[:3])

    # for i in range(2):
    #     sample_img = sample[i]['img']
    #     sample_mask = sample[i]['mask']
    #     print("------------transformed sample image, mask config-------------")
    #     print(f"[sample {i+1}]")
    #     print(sample_img.shape, sample_img.dtype, torch.min(sample_img), torch.max(sample_img))
    #     print(sample_mask.shape, sample_mask.dtype, torch.unique(sample_mask))
    #     print("--------------------------------------------------------------")
    
    train_ds = CacheDataset(
        data = train_dicts,
        transform = transforms,
        cache_num = 4,
        cache_rate = 1.0,
        num_workers = 0
        )

    val_ds = CacheDataset(
        data = val_dicts,
        transform = transforms,
        cache_num = 2,
        cache_rate = 1.0,
        num_workers = 0
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader