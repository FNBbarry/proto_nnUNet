from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

def combine_masks_to_multilabel_file(masks_dir, multilabel_file,class_map):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    masks_dir = Path(masks_dir)
    ref_img = nib.load(masks_dir / "liver.nii.gz")
    masks = class_map.keys()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx+1

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), multilabel_file)
if __name__ == '__main__':
    totalsegmentator_base = '/media/barry/SELF/DeepEyes_Project/Dataset/TotalSegmentator/Totalsegmentator_dataset_a_eval'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 507
    task_name = "TotalSegmentator"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    
    dataset_json_source = {}
    dataset_json_source['labels'] =  {
        "liver":1,
        "kidney_right":2,
        "kidney_left":3,
        "spleen":4,     
        "pancreas":5,
        "aorta":6,
        "inferior_vena_cava":7,   
        "adrenal_gland_right":8,
        "adrenal_gland_left":9,
        "gallbladder":10,
        "esophagus":11,
        "stomach":12, 
        "duodenum":13
        }
    dataset_csv_source = pd.read_csv(join(totalsegmentator_base, 'meta.csv'),sep=',')

    training_identifiers = list(dataset_csv_source[dataset_csv_source['split']=='train']['image_id'])

    for tr in training_identifiers:
        shutil.copy(join(totalsegmentator_base, tr,'ct.nii.gz'), join(imagestr, f'TotalSegmentator_{tr}_0000.nii.gz'))
        combine_masks_to_multilabel_file(join(totalsegmentator_base, tr,'segmentations'), join(labelstr, f'TotalSegmentator_{tr}.nii.gz'),dataset_json_source['labels'])
    
    test_identifiers = list(dataset_csv_source[dataset_csv_source['split']=='val']['image_id'])

    for ts in test_identifiers:
        shutil.copy(join(totalsegmentator_base, ts,'ct.nii.gz'), join(imagests, f'TotalSegmentator_{ts}_0000.nii.gz'))

    generate_dataset_json(out_base, 
                          {0:"CT"}, 
                          dataset_json_source['labels'],
                          num_training_cases=len(training_identifiers), 
                          file_ending='.nii.gz',
                          dataset_name=task_name)
