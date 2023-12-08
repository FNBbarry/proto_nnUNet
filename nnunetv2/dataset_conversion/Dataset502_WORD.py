from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

if __name__ == '__main__':
    word_base = '/media/barry/SELF/DeepEyes_Project/Dataset/WORD/WORD-V0.1.0'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 502
    task_name = "WORD"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    dataset_json_source = load_json(join(word_base, 'dataset.json'))

    dataset_json_source['labels'] = {v:int(k) for k,v in(( dataset_json_source['labels'].items()))}

    training_identifiers = [i['image'].split('/')[-1][:-7] for i in dataset_json_source['training']]

    for tr in training_identifiers:
        shutil.copy(join(word_base, 'imagesTr', tr + '.nii.gz'), join(imagestr, f'{tr}_0000.nii.gz'))
        shutil.copy(join(word_base, 'labelsTr', tr + '.nii.gz'), join(labelstr, f'{tr}.nii.gz'))

    test_identifiers = [i.split('/')[-1][:-7] for i in listdir(join(word_base,'imagesVal'))]

    for ts in test_identifiers:
        shutil.copy(join(word_base, 'imagesVal', ts + '.nii.gz'), join(imagests, f'{ts}_0000.nii.gz'))

    generate_dataset_json(out_base, 
                          {0:"CT"}, 
                          dataset_json_source['labels'],
                          num_training_cases=len(training_identifiers), 
                          file_ending='.nii.gz',
                          dataset_name=task_name)
