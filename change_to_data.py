from Brain_img.read_3D import get_data_loader_list
import torch
import os
import torchio as tio
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def tensor2nii(train_data_loader_list, val_data_loader_list):
    train_path_mri = "./nii_data/train/mri"
    if not os.path.exists(train_path_mri):
        os.makedirs(train_path_mri)

    train_path_pet = "./nii_data/train/pet"
    if not os.path.exists(train_path_pet):
        os.makedirs(train_path_pet)

    val_path_mri = "./nii_data/val/mri"
    if not os.path.exists(val_path_mri):
        os.makedirs(val_path_mri)

    val_path_pet = "./nii_data/val/pet"
    if not os.path.exists(val_path_pet):
        os.makedirs(val_path_pet)

    for i, data_it_train in enumerate(train_data_loader_list):
        device = torch.device("cuda:0")
        data = next(iter(data_it_train))
        img_mri = data['img_mri'].reshape(1, 48, 64, 48)
        img_pet = data['img_pet'].reshape(1, 48, 64, 48)
        mri_str_path = (train_path_mri + "/mri_train_{}_data.nii.gz").format(i)
        pet_str_path = (train_path_pet + "/pet_train_{}_data.nii.gz").format(i)
        image_mri = tio.ScalarImage(tensor=img_mri)
        image_pet = tio.ScalarImage(tensor=img_pet)
        image_mri.save(mri_str_path)
        image_pet.save(pet_str_path)

    for j, data_it_val in enumerate(val_data_loader_list):
        device = torch.device("cuda:0")
        data = next(iter(data_it_val))
        img_mri = data['img_mri'].reshape(1, 48, 64, 48).squeeze(-1)
        img_pet = data['img_pet'].reshape(1, 48, 64, 48).squeeze(-1)
        mri_str_path = (val_path_mri + "/mri_val_{}_data.nii.gz").format(j)
        pet_str_path = (val_path_pet + "/pet_val_{}_data.nii.gz").format(j)
        image_mri = tio.ScalarImage(tensor=img_mri)
        image_pet = tio.ScalarImage(tensor=img_pet)
        image_mri.save(mri_str_path)
        image_pet.save(pet_str_path)


def simple_show():
    mri_path = "./nii_data/train/mri/mri_train_0_data.nii.gz"
    image = nib.load(mri_path).get_fdata()
    print(image.shape)
    image = transforms.Grayscale(image[:, :, 2])
    plt.imshow(image)
    plt.show()


def main():
    train_file_list = ["./Brain_img/3D/train/train{}.tfrecords".format(i) for i in [i for i in range(73)]]
    train_file_indices = ["./Brain_img/index/train/train{}.index".format(i) for i in [i for i in range(73)]]
    val_file_list = ["./Brain_img/3D/validation/validation{}.tfrecords".format(i) for i in [i for i in range(8)]]
    val_file_indices = ["./Brain_img/index/validation/validation{}.index".format(i) for i in [i for i in range(8)]]
    train_data_loader_list, val_data_loader_list = get_data_loader_list(train_file_list, train_file_indices,
                                                                        val_file_list, val_file_indices, 73, 8)
    tensor2nii(train_data_loader_list, val_data_loader_list)
    #simple_show()


if __name__ == '__main__':
    main()

