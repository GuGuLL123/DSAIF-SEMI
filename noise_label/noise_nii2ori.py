import h5py
import numpy as np
import nibabel as nib
import os

dataname = 'Pancreas_CT'

if dataname == 'LA_DATASET':
    id_path = '/data/ylgu/Medical/Semi_medical_data/LA_DATASET/data/train.list'
    save_path = '/data2/ylgu/Medical/Semi_medical_data_noise/LA_DATASET/data/'
    image_path = '/data2/ylgu/Medical/Semi_medical_data/LA_DATASET_transform/imagesTr/'
    noise_label_path = '/data2/ylgu/Medical/Semi_medical_data/LA_DATASET_transform/sam3d/Semi_medical_data/LA_DATASET_transform/'
elif dataname == 'Pancreas_CT':
    id_path = '/data/ylgu/Medical/Semi_medical_data/Pancreas_CT/data/train.list'
    save_path = '/data2/ylgu/Medical/Semi_medical_data_noise/Pancreas_CT/data/'
    image_path = '/data2/ylgu/Medical/Semi_medical_data/Pancreas_CT_transform/imagesTr/'
    noise_label_path = '/data2/ylgu/Medical/Semi_medical_data/Pancreas_CT_transform/sam3d/Semi_medical_data/Pancreas_CT_transform/'
with open(id_path, 'r') as f:
    ids = f.read().splitlines()

for id in ids:
    print(id)
    if dataname == 'LA_DATASET':
        image_file = image_path + id + '.nii.gz'
        noise_label_file = noise_label_path + id + '_pred0.nii.gz'
        nifti_image = nib.load(image_file)
        nifti_label = nib.load(noise_label_file)
        image_data = nifti_image.get_fdata()
        label_data = nifti_label.get_fdata()
        # print(np.max(image_data))
        # print(np.max(label_data))
        save_name = save_path + '2018LA_Seg_Training Set/' + id + '/' + 'mri_norm2.h5'
        if not os.path.exists(save_path + '2018LA_Seg_Training Set/' + id):
            os.makedirs(save_path + '2018LA_Seg_Training Set/' + id)

        with h5py.File(save_name, 'w') as f:
            f.create_dataset('image', data=image_data)
            f.create_dataset('label', data=label_data)
        print('1')

    if dataname == 'Pancreas_CT':
        image_file = image_path + id + '.nii.gz'
        noise_label_file = noise_label_path + id + '_pred0.nii.gz'
        nifti_image = nib.load(image_file)
        nifti_label = nib.load(noise_label_file)
        image_data = nifti_image.get_fdata()
        label_data = nifti_label.get_fdata()
        # print(np.max(image_data))
        # print(np.max(label_data))
        save_name = save_path + 'Pancreas_h5/' + id + '_norm.h5'
        if not os.path.exists(save_path + 'Pancreas_h5'):
            os.makedirs(save_path + 'Pancreas_h5')
        with h5py.File(save_name, 'w') as f:
            f.create_dataset('image', data=image_data)
            f.create_dataset('label', data=label_data)
        print('2')