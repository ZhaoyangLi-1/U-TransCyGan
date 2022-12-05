from Brain_img.read_3D import get_data_loader_list
import torch
import os
import torchio as tio
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Brain_img.read_3D import get_data_loader
import argparse


def tensor2nii(train_loader, val_loader):
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
    
    for i, data in enumerate(train_loader):
        img_mri = data['img_mri'].reshape(1, 48, 64, 48)
        img_pet = data['img_pet'].reshape(1, 48, 64, 48)
        mri_str_path = (train_path_mri + "/mri_train_{}_data.nii.gz").format(i)
        pet_str_path = (train_path_pet + "/pet_train_{}_data.nii.gz").format(i)
        image_mri = tio.ScalarImage(tensor=img_mri)
        image_pet = tio.ScalarImage(tensor=img_pet)
        image_mri.save(mri_str_path)
        image_pet.save(pet_str_path)

    for j, data in enumerate(val_loader):
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


def main(args):
    train_loader, val_loader = get_data_loader(args)
    print("Begin Save")
    tensor2nii(train_loader, val_loader)
    print("Finished Saved")

    #simple_show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Basic options
    parser.add_argument('--train_tfrecord_pattern', default='./Brain_img/3D/train/train{}.tfrecords', type=str,
                        help='train tfrecord path to a batch of content and style images')
    parser.add_argument('--train_index_pattern', default='./Brain_img/index/train/train{}.index', type=str,
                        help='train index pattern path to a batch of contecnt and style images')
    parser.add_argument('--val_tfrecord_pattern', default='./Brain_img/3D/validation/validation{}.tfrecords', type=str,
                        help='validation tfrecord path to a batch of validation data')
    parser.add_argument('--val_index_pattern', default='./Brain_img/index/validation/validation{}.index', type=str,
                        help='validation index pattern to a batch of validation data')
    parser.add_argument('--n_train', default=726, type=int)
    parser.add_argument('--n_val', default=80, type=int)

    parser.add_argument('--input_w', default=96, type=int)
    parser.add_argument('--input_h', default=128, type=int)
    parser.add_argument('--input_d', default=96, type=int)
    parser.add_argument('--resume_path', default='./pretrained_weights/resnet_50_epoch_110_batch_0.pth.tar')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--pretrained_model', default='./pretrained_weights/resnet_50.pth', type=str)
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')

    # train options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4) 
    parser.add_argument('--lamda_cycle', type=float, default=10)
    parser.add_argument('--lamda_identity', type=float, default=0.5)
    parser.add_argument('--checkpoint_gen_m', default='./saved/genm.pth.tar')
    parser.add_argument('--checkpoint_gen_p', default='./saved/genp.pth.tar')
    parser.add_argument('--checkpoint_disc_m', default='./saved/discm.pth.tar')
    parser.add_argument('--checkpoint_disc_p', default='./saved/discp.pth.tar')
    parser.add_argument('--wgan', action='store_true', help='use Wasserstein GAN for training instead of classic GAN')
    parser.add_argument('--wgan_n_critic', type=int, default=5, help='number of iterations for training the wgan critic before training the wgan generator')
    parser.add_argument('--wgan_clamp_lower', type=float, default=-0.01, help='lower bound for wgan Lipschitz clamping')
    parser.add_argument('--wgan_clamp_upper', type=float, default=0.01, help='upper bound for wgan Lipschitz clamping')
    parser.add_argument('--wgan_lrD', type=float, default=0.00005, help='learning rate for wgan Critic, default=0.00005')
    parser.add_argument('--wgan_lrG', type=float, default=0.00005, help='learning rate for wgan Generator, default=0.00005')
    parser.add_argument('--wgan_optimizer', type=str, default='adam', help='optimizer for the generator and discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lamdar_gp', type=float, default=10, help='Gradient penalty lambda hyperparameter')
    parser.add_argument('--prin_frep', type=float, default=20, help='how much frequence we want to print')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--l1_lamda', type=int, default=0.01, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--diff_aug', type=str, default="interpolation,cutout,translation",)
    parser.add_argument('--is_pretrained', type=bool, default=False, help='check to use check point')
    args = parser.parse_args()

    main(args)


