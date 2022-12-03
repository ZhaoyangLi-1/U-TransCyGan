#from this import d
from pickle import FALSE
#from this import d
#from time import sleep
from tkinter import image_names
import torch
from torch import fake_quantize_per_tensor_affine, nn
import torch.nn.functional as F
import os
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
#from torchvision.utils import save_image
import SimpleITK as sitk
from GPUtil import showUtilization as gpu_usage
import torchio as tio
import numpy as np
from Brain_img.read_3D import get_data_loader
from models.generator import generate_ConvTrantGe
from models.discriminator import generate_dis
#from models.TransGAN_Disc import generate_dis
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from models.GANLoss import GANLoss
from models.GANLoss import cal_gradient_penalty, cal_penalty_intensity
from itertools import repeat
from utils.utils import ImagePool
from torch.optim import lr_scheduler
#from Brain_img_V2.get_data_loader import get_data_loader

GRADIENT_ACCUMULATION = 2

BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter



def show_loss_graph(epoch_array, loss_Array, file_pth, title):
    plt.figure(figsize=(12,8))
    plt.plot(epoch_array, loss_Array, marker='o', label='D_loss', color='r')
    plt.xlabel("Step")
    plt.xlabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(file_pth)
    plt.show

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()
    torch.cuda.empty_cache()

    print("GPU Usage after emptying the cache")
    gpu_usage()


def adjust_learning_rate(iteration_count, args, opt_gen, opt_disc_m, opt_disc_p):
    """Imitating the original implementation"""
    args.lr = 1.2 - max(0, iteration_count + args.epoch_count  - args.max_iter) / float(args.n_epochs_decay + 1)
    #args.lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in opt_disc_m.param_groups:
        param_group['lr'] = args.lr
    for param_group in opt_disc_p.param_groups:
        param_group['lr'] = args.lr
    for param_group in opt_gen.param_groups:
        param_group['lr'] = args.lr
    


def warmup_learning_rate(iteration_count, args,  opt_gen, opt_disc_m, opt_disc_p):
    """Imitating the original implementation"""
    #args.lr =1.2 - max(0, iteration_count + args.epoch_count  - args.max_iter) / float(args.n_epochs_decay + 1)
    args.lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in opt_disc_m.param_groups:
        param_group['lr'] = args.lr
    for param_group in opt_disc_p.param_groups:
        param_group['lr'] = args.lr
    for param_group in opt_gen.param_groups:
        param_group['lr'] = args.lr
    # print(lr)

def set_grad(D_P, D_M, is_grad):
    for p in D_M.parameters():
        p.requires_grad = is_grad  # to avoid computation

    for p in D_P.parameters():
        p.requires_grad = is_grad  # to avoid computation


def compute_regular(model, lamda):
    all_params = torch.cat([x.view(-1) for x in model.parameters()])
    return 0.5 * lamda * torch.norm(all_params, 2)


def train(train_loader, args, G_M, G_P, D_P, D_M, L1, criterionGAN,
          opt_gen, opt_disc, device, writer, G_loss_array, 
          D_loss_mri_array, D_loss_pet_array, epoch_array):
    scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=100)
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=100)
    # USE_CUDA = torch.cuda.is_available()
    show_step = 0
    # mri_pool = ImagePool(args.batch_size)
    # pet_pool = ImagePool(args.batch_size)
    for epoch in range(args.max_iter):
        # if epoch < 1e4:
        #      warmup_learning_rate(iteration_count=epoch, args=args, opt_gen=opt_gen, opt_disc_m=opt_dis_m, opt_disc_p=opt_dis_p)
        # else:
        #     adjust_learning_rate(iteration_count=epoch, args=args, opt_gen=opt_gen, opt_disc_m=opt_dis_m, opt_disc_p=opt_dis_p)
        print("Learning rate is {}".format(opt_gen.param_groups[0]['lr']))

        for batch_idx, batch in enumerate(tqdm(list(train_loader))):
            #sleep(0.01)
            #total_step += args.batch_size
            gender = batch['gender']
            age = batch['age']
            img_mri = batch['img_mri'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3).to(device)
            img_pet = batch['img_pet'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3).to(device)
            # img_mri = F.interpolate(img_mri_orin, size=(96, 128, 96), mode='nearest').to(device)
            # img_pet = F.interpolate(img_pet_orin, size=(96, 128, 96), mode='nearest').to(device)

        #     ###### Generators A2B and B2A ###### 
            # set_grad(D_M, D_P, False)
            # opt_gen.zero_grad()
            # # Identity loss
            # same_pet = G_P(img_pet)
            # loss_identity_pet = L1(same_pet, img_pet) * args.lamda_identity
            # same_mri = G_M(img_mri)
            # loss_identity_mri = L1(same_mri, img_mri) * args.lamda_identity
            # del same_pet, same_mri

            # # GAN loss
            # fake_mri = G_M(img_pet)
            # pred_fake = D_M(fake_mri)
            # loss_GAN_p2m = criterionGAN(pred_fake, True)

            # fake_pet = G_P(img_mri)
            # pred_fake = D_P(fake_pet)
            # loss_GAN_m2p = criterionGAN(pred_fake, True)

            # # cycle loss
            # recovered_mri = G_M(fake_pet)
            # loss_cycle_mpm = L1(recovered_mri, img_mri) * args.lamda_cycle
            # recovered_pet = G_P(fake_mri)
            # loss_cycle_pmp = L1(recovered_pet, img_pet) * args.lamda_cycle
            # del recovered_mri, recovered_pet

            # G_loss = loss_identity_pet + loss_identity_mri + loss_GAN_p2m + loss_GAN_m2p + loss_cycle_mpm + loss_cycle_pmp
            # G_loss.backward()
            # opt_gen.step()

        #     set_grad(D_M, D_P, True)
        #     ###### Discriminator MRI ######
        #     for i in range(args.wgan_n_critic):
        #         # Real loss
        #         opt_dis_m.zero_grad() 
        #         pred_real = D_M(img_mri)
        #         loss_real = criterionGAN(pred_real, True)
        #         # Fake loss
        #         fake_mri = mri_pool.query(fake_mri)
        #         pred_fake = D_M(fake_mri.detach())
        #         loss_fake = criterionGAN(pred_fake, False)
        #         gp = cal_gradient_penalty(D_M, img_mri, fake_mri, device)
        #         # Total loss
        #         loss_D_M = loss_real + loss_fake + gp
        #         loss_D_M.backward()
        #         opt_dis_m.step()

        #     for i in range(args.wgan_n_critic):
        #         # Real loss
        #         opt_dis_p.zero_grad() 
        #         pred_real = D_P(img_pet)
        #         loss_real = criterionGAN(pred_real, True)
        #         # Fake loss
        #         fake_pet = pet_pool.query(fake_pet)
        #         pred_fake = D_P(fake_pet.detach())
        #         loss_fake = criterionGAN(pred_fake, False)
        #         gp = cal_gradient_penalty(D_P, img_pet, fake_pet, device)

        #         # Total loss
        #         loss_D_P = loss_real + loss_fake + gp
        #         loss_D_P.backward()
        #         opt_dis_p.step()
            
        #     # update learning rate
        #     scheduler_gen.step()
        #     scheduler_dis_p.step()
        #     scheduler_dis_m.step()

            set_grad(D_M, D_P, True)
            opt_disc.zero_grad() 
            fake_pet = G_P(img_mri)
            D_P_real = D_P(img_pet)
            D_P_fake = D_P(fake_pet.detach())
            D_P_real_loss = criterionGAN(D_P_real, True) # L1
            D_P_fake_loss = criterionGAN(D_P_fake, False) # L1 
            gp = cal_gradient_penalty(D_P, img_pet, fake_pet, device, lambda_gp=10)
            lambda_intensity = cal_penalty_intensity(fake_pet)
            # l1_regular_P = compute_regular(D_P, args.l1_lamda)
            D_P_loss = D_P_real_loss + lambda_intensity*D_P_fake_loss + gp

        
            fake_mri = G_M(img_pet)
            D_M_real = D_M(img_mri)
            D_M_fake = D_M(fake_mri.detach())
            D_M_real_loss = criterionGAN(D_M_real,True)
            D_M_fake_loss = criterionGAN(D_M_fake, False)
            gp = cal_gradient_penalty(D_M, img_mri, fake_mri, device, lambda_gp=10)
            #l1_regular_M = compute_regular(D_M, args.l1_lamda)
            lambda_intensity = cal_penalty_intensity(fake_mri)
            D_M_loss = D_M_real_loss + lambda_intensity*D_M_fake_loss + gp

            D_loss = (D_P_loss + D_M_loss) * 0.5 #+ (l1_regular_P + l1_regular_M)
            D_loss.backward()
            opt_disc.step()

            if batch_idx % args.wgan_n_critic:
                set_grad(D_M, D_P, False)
                opt_gen.zero_grad()
                # Identity loss
                same_pet = G_P(img_pet)
                loss_identity_pet = L1(same_pet, img_pet) * args.lamda_identity
                same_mri = G_M(img_mri)
                loss_identity_mri = L1(same_mri, img_mri) * args.lamda_identity
                del same_pet, same_mri

                # GAN loss
                fake_mri = G_M(img_pet)
                pred_fake = D_M(fake_mri)
                loss_GAN_p2m = criterionGAN(pred_fake, True)

                fake_pet = G_P(img_mri)
                pred_fake = D_P(fake_pet)
                loss_GAN_m2p = criterionGAN(pred_fake, True)

                # cycle loss
                recovered_mri = G_M(fake_pet)
                loss_cycle_mpm = L1(recovered_mri, img_mri) * args.lamda_cycle
                recovered_pet = G_P(fake_mri)
                loss_cycle_pmp = L1(recovered_pet, img_pet) * args.lamda_cycle
                del recovered_mri, recovered_pet

                #l1_regular_M = compute_regular(G_M, args.l1_lamda)
                #l2_regular_P = compute_regular(G_P, args.l1_lamda)
                #l1_regular = l1_regular_M + l2_regular_P

                G_loss = (loss_identity_pet + 
                        loss_identity_mri + 
                        loss_GAN_p2m + 
                        loss_GAN_m2p + 
                        loss_cycle_mpm + 
                        loss_cycle_pmp)
                        #l1_regular)

                G_loss.backward()
                opt_gen.step()

            
                show_step += 1
                writer.add_scalar('Generator_Loss', G_loss.item(), epoch+1, batch_idx+1)
                writer.add_scalar('Discriminator_PET_Loss', D_M_loss.item(), epoch+1, batch_idx+1)
                writer.add_scalar('Discriminator_MRI_Loss', D_P_loss.item(), epoch+1, batch_idx+1)

                for i in range(args.batch_size):
                    mri_str_path = ("./train_data/mri" + "/mri_train_batch{}_data_{}.nii.gz").format(batch_idx+1, i+1)
                    pet_str_path = ("./train_data/pet" + "/pet_train_batch{}_data_{}.nii.gz").format(batch_idx+1, i+1)
                    # fake_mri = F.interpolate(fake_mri, size=(48, 64, 48), mode='nearest')
                    # fake_pet = F.interpolate(fake_pet, size=(48, 64, 48), mode='nearest')
                    tran_mri = tio.ScalarImage(tensor=fake_mri[i, :, :, :, :].type(torch.float32).cpu().detach().numpy())
                    tran_pet = tio.ScalarImage(tensor=fake_pet[i, :, :, :, :].type(torch.float32).cpu().detach().numpy())
                    tran_mri.save(mri_str_path)
                    tran_pet.save(pet_str_path)

                # for i in range(args.batch_size):
                #         mri_str_path = ("./train_data/new_mri" + "/mri_train_batch{}_data_{}.nii.gz").format(batch_idx+1, i+1)
                #         pet_str_path = ("./train_data/new_pet" + "/pet_train_batch{}_data_{}.nii.gz").format(batch_idx+1, i+1)
                #         fake_mri = F.interpolate(fake_mri, size=(48, 64, 48), mode='nearest')
                #         fake_pet = F.interpolate(fake_pet, size=(48, 64, 48), mode='nearest')
                #         image_mri = tio.ScalarImage(tensor=fake_mri[i, :, :, :, :].type(torch.float32).cpu().detach().numpy())
                #         image_pet = tio.ScalarImage(tensor=fake_pet[i, :, :, :, :].type(torch.float32).cpu().detach().numpy())
                #         image_mri.save(mri_str_path)
                #         image_pet.save(pet_str_path)

                print(
                "[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [D MRI Loss: %f] [D PET Loss: %f] [G loss: %f]"
                % (epoch+1, args.max_iter, batch_idx+1, len(list(train_loader)), D_loss.item(), D_M_loss.item(), D_P_loss.item(), G_loss.item())
                )
                G_loss_array.append(G_loss.item())
                D_loss_mri_array.append(D_M_loss.item())
                D_loss_pet_array.append(D_P_loss.item())
                epoch_array.append(show_step)
         
        scheduler_gen.step()
        scheduler_disc.step()
                
            
            
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def main(args):

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    gpus = [0,1,2,3,4,5,6,7]
    #gpus = [1,2,3]

    G_M = generate_ConvTrantGe(args)
    for module in G_M.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    G_P = generate_ConvTrantGe(args)
    for module in G_P.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    D_M = generate_dis(args)
    D_P = generate_dis(args)

    G_M = nn.DataParallel(G_M, device_ids=gpus, output_device=gpus[0])
    G_P = nn.DataParallel(G_P, device_ids=gpus, output_device=gpus[0])
    D_M = nn.DataParallel(D_M, device_ids=gpus, output_device=gpus[0])
    D_P = nn.DataParallel(D_P, device_ids=gpus, output_device=gpus[0])

    G_M.to(device)
    G_P.to(device)
    D_M.to(device)
    D_P.to(device)
  
    opt_gen = optim.Adam(
        list(G_M.parameters()) + list(G_P.parameters()),
        lr=args.lr,
        betas=(0.5, 0.99),
    )

    opt_disc = optim.RMSprop(
        list(D_M.parameters()) + list(D_P.parameters()),
        lr=args.lr,
        alpha=0.9,
    )

    # opt_dis_m = optim.Adam(
    #     D_M.parameters(),
    #     lr=args.lr,
    #     betas=(0.5, 0.99),
    # )

    # opt_dis_p = optim.Adam(
    #     D_P.parameters(),
    #     lr=args.lr,
    #     betas=(0.5, 0.99),
    # )

    L1 = nn.L1Loss()
    criterionGAN = GANLoss('wgangp')

    train_loader, _ = get_data_loader(args)

    G_loss_array=[]
    D_loss_mri_array=[]
    D_loss_pet_array=[]

    epoch_array =[]

    train(train_loader, args, G_M, G_P, D_P, D_M, L1, criterionGAN,
          opt_gen, opt_disc, device, writer, G_loss_array, 
          D_loss_mri_array, D_loss_pet_array, epoch_array)

    #epoch_array = np.array(epoch_array)
    epoch_array = np.array(epoch_array)
    G_loss_array = np.array(G_loss_array)
    D_loss_mri_array =np.array(D_loss_mri_array)
    D_loss_pet_array =np.array(D_loss_pet_array)
    show_loss_graph(epoch_array, G_loss_array, "./Generator_loss.jpg", "Generator Loss")
    show_loss_graph(epoch_array, D_loss_mri_array, "./Disc_mri_loss.jpg", "Discriminator MRI Loss")
    show_loss_graph(epoch_array, D_loss_pet_array, "./Disc_pet_loss.jpg", "Discriminator PET Loss")
    
    save_checkpoint(G_M, opt_gen, filename=args.checkpoint_gen_m)
    save_checkpoint(G_P, opt_gen, filename=args.checkpoint_gen_p)
    save_checkpoint(D_P, opt_disc, filename=args.checkpoint_disc_p)
    save_checkpoint(D_M, opt_disc, filename=args.checkpoint_disc_m)
    # save_checkpoint(G_M, opt_gen, filename='./saved/genm_new.pth.tar')
    # save_checkpoint(G_P, opt_gen, filename='./saved/genp_new.pth.tar')
    # save_checkpoint(D_P, opt_disc, filename='./saved/discm_new.pth.tar')
    # save_checkpoint(D_M, opt_disc, filename='./saved/discp_new.pth.tar')


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
    parser.add_argument('--n_train', default=73, type=int)
    parser.add_argument('--n_val', default=8, type=int)

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
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8) 
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
    parser.add_argument('--prin_frep', type=float, default=5, help='how much frequence we want to print')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--l1_lamda', type=int, default=0.01, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--diff_aug', type=str, default="interpolation,cutout,translation", help='Data Augmentation')
    args = parser.parse_args()

    main(args)
