from PIL import Image
import torch
import torchio as tio
from torch.utils.data import DataLoader
import torch.nn.functional as F

from test import main


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, mri_txt, pet_txt, transform=None, target_transform=None):

        fh_mri = open(mri_txt, 'r')
        mri_img = []
        fh_pet = open(pet_txt, 'r')
        pet_img = []
        for line in fh_mri:
            line = line.rstrip()
            words = line.split()
            mri_img.append(words)
        for line in fh_pet:
            line = line.rstrip()
            words = line.split()
            pet_img.append(words)
        self.mri_img = mri_img
        self.pet_img = pet_img
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

    def __getitem__(self, index):
        fn_mri = self.mri_img[index][0]
        fn_pet = self.pet_img[index][0]
        img_mri = tio.ScalarImage(self.root+fn_mri).data.type(torch.float32)
        img_pet = tio.ScalarImage(self.root+fn_pet).data.type(torch.float32)

        if self.transform is not None:
            img_mri = self.transform(img_mri)
            img_pet = self.transform(img_pet)

        return img_mri, img_pet

    def __len__(self):
        return len(self.mri_img)


def get_data_loader(ags):
    root = "./Brain_img_V2/train/"
    train_data = MyDataset(root=root, mri_txt='./Brain_img_V2/train_mri_paths.txt', pet_txt='./Brain_img_V2/train_pet_paths.txt')
    train_loader = DataLoader(dataset=train_data, batch_size=7, shuffle=True)
    return train_loader
