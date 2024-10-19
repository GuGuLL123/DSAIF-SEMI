import h5py
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

dataname = 'Pancreas_CT'

if dataname == 'LA_DATASET':
    id_path = '/data/ylgu/Medical/Semi_medical_data/LA_DATASET/data/train.list'
    save_path = '/data2/ylgu/Medical/Semi_medical_data/LA_DATASET_transform/'
elif dataname == 'Pancreas_CT':
    id_path = '/data/ylgu/Medical/Semi_medical_data/Pancreas_CT/data/train.list'
    save_path = '/data2/ylgu/Medical/Semi_medical_data/Pancreas_CT_transform/'
with open(id_path, 'r') as f:
    ids = f.read().splitlines()

for id in ids:
    print(id)
    if dataname == 'LA_DATASET':
        output_size = (128, 128, 128)
        h5f = h5py.File(
            "/data/ylgu/Medical/Semi_medical_data/LA_DATASET/data/2018LA_Seg_Training Set/" + id + "/mri_norm2.h5", 'r')
    elif dataname == 'Pancreas_CT':
        output_size = (128, 128, 128)
        h5f = h5py.File("/data/ylgu/Medical/Semi_medical_data/Pancreas_CT/data/Pancreas_h5/" + id + "_norm.h5", 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    # zoom_factors = [n / o for n, o in zip(output_size, image.shape)]
    # image = zoom(image, zoom_factors,order=1)
    # label = zoom(label, zoom_factors,order=0)

    
    # if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
    #     pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
    #     ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
    #     pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
    #     image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
    #     label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    # (w, h, d) = image.shape

    # w1 = np.random.randint(0, w - output_size[0])
    # h1 = np.random.randint(0, h - output_size[1])
    # d1 = np.random.randint(0, d - output_size[2])

    # label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    # image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

    print(image.shape)
    print(label.shape)

    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
    nifti_label = nib.Nifti1Image(label, affine=np.eye(4))

    image_save_path = save_path + 'imagesTr/' + id + '.nii.gz'
    label_save_path = save_path + 'labelsTr/' + id + '.nii.gz'
    nib.save(nifti_image, image_save_path)
    nib.save(nifti_label, label_save_path)

