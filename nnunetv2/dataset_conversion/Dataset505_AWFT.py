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
    ref_img = nib.load(masks_dir / "1.nii")
    masks = class_map.values()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if mask == 0:
            continue
        if os.path.exists(f"{masks_dir}/{mask}.nii"):
            img = nib.load(f"{masks_dir}/{mask}.nii").get_fdata()
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), multilabel_file)

def split_multilabel_file_to_masks(multilabel_file,masks_dir,class_map,convert_names=None,skip_class=[]):
    ref_img = nib.load(multilabel_file)
    masks_dir = Path(masks_dir)
    img = ref_img.get_fdata()
    
    for mask in class_map.values():
        if mask == 0:
            continue
        if mask in skip_class:
            continue
        converted_data = np.zeros_like(img)
        converted_data[img == int(mask)] = 1
        if convert_names == None:
            nib.save(nib.Nifti1Image(converted_data, ref_img.affine), join(masks_dir,str(mask)))
        else:
            nib.save(nib.Nifti1Image(converted_data, ref_img.affine), join(masks_dir,str(convert_names[mask])))
if __name__ == '__main__':
    dataset_base = '/media/barry/SELF/DeepEyes_Project/nnUNetFrame/DATASET/nnUNet_raw'
    Datasets = ['Dataset501_AMOS22','Dataset502_WORD','Dataset503_FLARE2022','Dataset504_TotalSegmentator']
    # Datasets = ['Dataset504_TotalSegmentator']

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 505
    task_name = "AWFT"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    
    dataset_json_source_all = {}
    dataset_json_source_all['labels'] =  {
        "background": 0,
        "liver":1,
        "kidney_right":2,
        "kidney_left":3,
        "spleen":4,     
        "pancreas":5,
        # "aorta":6,
        # "inferior_vena_cava":7,   
        # "adrenal_gland_right":8,
        # "adrenal_gland_left":9,
        "gallbladder":6,
        "esophagus":7,
        "stomach":8, 
        "duodenum":9
        }

    for dataset in Datasets:
        dataset_json_source = load_json(join(dataset_base, dataset,'dataset.json'))
        convert_names = {}
        if dataset == 'Dataset501_AMOS22':
            convert_names[1] = 4
            convert_names[2] = 2
            convert_names[3] = 3
            convert_names[4] = 6
            convert_names[5] = 7
            convert_names[6] = 1
            convert_names[7] = 8
            convert_names[10]= 5
            convert_names[13] = 9
        elif dataset == 'Dataset502_WORD':
            convert_names[1] = 1
            convert_names[2] = 4
            convert_names[3] = 3
            convert_names[4] = 2
            convert_names[5] = 8
            convert_names[6] = 6
            convert_names[7] = 7
            convert_names[8] = 5
            convert_names[9] = 9
        elif dataset == 'Dataset503_FLARE2022':
            convert_names[1] = 1
            convert_names[2] = 2
            convert_names[3] = 4
            convert_names[4] = 5
            convert_names[9] = 6
            convert_names[10] = 7
            convert_names[11] = 8
            convert_names[12] = 9
            convert_names[13] = 3
        elif dataset == 'Dataset504_TotalSegmentator':
            convert_names[1] = 1
            convert_names[2] = 2
            convert_names[3] = 3
            convert_names[4] = 4
            convert_names[5] = 5
            convert_names[10] = 6
            convert_names[11] = 7
            convert_names[12] = 8
            convert_names[13] = 9

        for tr in os.listdir(join(dataset_base, dataset,'imagesTr')):
            print(tr)
            tmp_one_labels = '/media/barry/SELF/DeepEyes_Project/20230917/tmp_one_labels/'
            maybe_mkdir_p(tmp_one_labels)
            shutil.copy(join(dataset_base, dataset,'imagesTr', tr), join(imagestr, tr[:-12]+'_0000.nii.gz'))
            if dataset == 'Dataset501_AMOS22':
                split_multilabel_file_to_masks(join(dataset_base, dataset,'labelsTr', tr[:-12]+'.nii.gz'),tmp_one_labels,dataset_json_source['labels'],skip_class=[8,9,11,12,14,15],convert_names=convert_names)
            elif dataset == 'Dataset502_WORD':
                split_multilabel_file_to_masks(join(dataset_base, dataset,'labelsTr', tr[:-12]+'.nii.gz'),tmp_one_labels,dataset_json_source['labels'],skip_class=[10,11,12,13,14,15,16],convert_names=convert_names)
            elif dataset == 'Dataset503_FLARE2022':
                split_multilabel_file_to_masks(join(dataset_base, dataset,'labelsTr', tr[:-12]+'.nii.gz'),tmp_one_labels,dataset_json_source['labels'],skip_class=[5,6,7,8],convert_names=convert_names)
            elif dataset == 'Dataset504_TotalSegmentator':
                split_multilabel_file_to_masks(join(dataset_base, dataset,'labelsTr', tr[:-12]+'.nii.gz'),tmp_one_labels,dataset_json_source['labels'],skip_class=[6,7,8,9],convert_names=convert_names)
            
            combine_masks_to_multilabel_file(tmp_one_labels,join(labelstr, tr[:-12]+'.nii.gz'),dataset_json_source_all['labels'])
            shutil.rmtree(tmp_one_labels)

        for ts in os.listdir(join(dataset_base, dataset,'imagesTs')):
            shutil.copy(join(dataset_base,dataset, 'imagesTs', ts), join(imagests, ts[:-12]+'_0000.nii.gz'))
    
    generate_dataset_json(out_base, 
                        {0:"CT"}, 
                        dataset_json_source_all['labels'],
                        num_training_cases=len(os.listdir(imagestr)), 
                        file_ending='.nii.gz',
                        dataset_name=task_name)
