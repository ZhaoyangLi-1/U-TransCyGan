from tkinter import image_names
import torch
from torch import nn
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
from utils.utils import ImagePool
#from Brain_img_V2.get_data_loader import get_data_loader

GRADIENT_ACCUMULATION = 3

BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter


def show_loss_graph(epoch_array, loss_Array, file_pth, title):
    plt.figure(figsize=(12,8))
    plt.plot(epoch_array, loss_Array, marker='o', label='D_loss', color='r')
    plt.xlabel("Epoch")
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


def adjust_learning_rate(iteration_count, args, opt_disc, opt_gen):
    """Imitating the original implementation"""
    args.lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    #args.lr = 2e-4 - max(0, iteration_count + args.epoch_count - args.n_epochs) / float(args.n_epochs_decay + 1)
    for param_group in opt_disc.param_groups:
        param_group['lr'] = args.lr
    for param_group in opt_gen.param_groups:
        param_group['lr'] = args.lr
    


def warmup_learning_rate(iteration_count, args, opt_disc, opt_gen):
    """Imitating the original implementation"""
    args.lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in opt_disc.param_groups:
        param_group['lr'] = args.lr
    for param_group in opt_gen.param_groups:
        param_group['lr'] = args.lr
    


def train(train_loader, args, G_M, G_P, D_P, D_M, L1, criterionGAN, opt_disc, opt_gen, g_scaler, d_scaler, device, writer, epoch, fake_mri_pool, fake_pet_pool):
    loop = tqdm(train_loader, leave=True)
    loop.set_postfix
    #print(out.shape)

    #USE_CUDA = torch.cuda.is_available()
    #device = torch.device("cuda:0" if USE_CUDA else "cpu")
    #batch_idx = 0
    G_loss_total=torch.zeros(1).to(device)
    D_M_loss_total=torch.zeros(1).to(device)
    D_P_loss_total=torch.zeros(1).to(device)
    G_GAN_Loss_total=torch.zeros(1).to(device)

    batch_idx=0
    for batch_idx, batch in enumerate(train_loader):
        img_mri = batch['img_mri'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3)
        img_pet = batch['img_pet'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3)
        #N, C, D, H, W = content_images.shape
        img_mri = F.interpolate(img_mri, size=(96, 128, 96), mode='nearest').to(device)
        img_pet = F.interpolate(img_pet, size=(96, 128, 96), mode='nearest').to(device)
        #mri_pet = torch.stack((image_mri, img_pet), 0)
        #del img_mri, img_pet


        # Train Generator
        for p in D_M.parameters():
            p.requires_grad = False  # to avoid computation

        for p in D_P.parameters():
            p.requires_grad = False  # to avoid computation
        
        opt_gen.zero_grad()
        #with torch.cuda.amp.autocast():
        # adversarial loss for both generators
        fake_pet = G_P(img_mri)
        fake_mri = G_M(img_pet)

        D_P_fake = D_P(fake_pet)
        D_M_fake = D_M(fake_mri)
        loss_G_M = criterionGAN(D_M_fake, torch.ones_like(D_M_fake))
        loss_G_P = criterionGAN(D_P_fake, torch.ones_like(D_P_fake))

        # cycle loss
        cycle_mri = G_M(fake_pet)
        cycle_mri_loss = L1(img_mri, cycle_mri) * args.lamda_identity
        del cycle_mri

        cycle_pet = G_P(fake_mri)
        cycle_pet_loss = L1(img_pet, cycle_pet) * args.lamda_identity
        del cycle_pet

        # identity loss (remove these for efficiency if you set lambda_identity=0)
        identity_mri = G_M(img_mri)
        identity_mri_loss = L1(img_mri, identity_mri)* args.lamda_cycle
        del identity_mri

        identity_pet = G_P(img_pet)
        identity_pet_loss = L1(img_pet, identity_pet)* args.lamda_cycle
        del identity_pet
        G_GAN_Loss_total += loss_G_P + loss_G_M
        G_loss = (   # six loss 
            loss_G_P
            + loss_G_M
            + cycle_mri_loss 
            + cycle_pet_loss
            + identity_mri_loss
            + identity_pet_loss
        )
        G_loss_total += G_loss
        G_loss.backward()
        opt_gen.step()

        for p in D_P.parameters():
            p.requires_grad = True
        
        for p in D_M.parameters():
            p.requires_grad = True
        
        opt_disc.zero_grad()

        #with torch.cuda.amp.autocast():
        fake_pet = fake_pet_pool.query(fake_pet)
        D_P_real = D_P(img_pet)
        D_P_fake = D_P(fake_pet.detach())
        D_P_real_loss = criterionGAN(D_P_real,  torch.ones_like(D_P_real))
        D_P_fake_loss = criterionGAN(D_P_fake,  torch.zeros_like(D_P_fake))
        D_P_loss = (D_P_real_loss + D_P_fake_loss)/2
        D_P_loss_total += D_P_loss
        
        D_P_loss.backward()

        #with torch.cuda.amp.autocast():
        fake_mri = fake_mri_pool.query(fake_mri)
        D_M_real = D_M(img_mri)
        D_M_fake = D_M(fake_mri.detach())
        D_M_real_loss = criterionGAN(D_M_real, torch.ones_like(D_M_real))
        D_M_fake_loss = criterionGAN(D_M_fake, torch.zeros_like(D_M_fake))
        D_M_loss = (D_M_real_loss + D_M_fake_loss)/2
        D_M_loss_total += D_M_loss

        D_M_loss.backward()

        opt_disc.step()
        
        D_loss = D_M_loss+D_P_loss
        

        # writer.add_scalar('Generator_Loss', G_loss.sum().item(), epoch + 1, batch_idx+1)
        # writer.add_scalar('Discriminator_Loss', D_loss.sum().item(), epoch + 1, batch_idx+1)
        #batch_idx = batch_idx + 1
        if batch_idx % args.prin_frep == 0:
            for i in range(args.batch_size):
                        mri_str_path = ("./train_data/new_mri" + "/mri_train_batch{}_data_{}.nii.gz").format(batch_idx+1, i+1)
                        pet_str_path = ("./train_data/new_pet" + "/pet_train_batch{}_data_{}.nii.gz").format(batch_idx+1, i+1)
                        fake_mri = F.interpolate(fake_mri, size=(48, 64, 48), mode='nearest')
                        fake_pet = F.interpolate(fake_pet, size=(48, 64, 48), mode='nearest')
                        image_mri = tio.ScalarImage(tensor=fake_mri[i, :, :, :, :].type(torch.float32).cpu().detach().numpy())
                        image_pet = tio.ScalarImage(tensor=fake_pet[i, :, :, :, :].type(torch.float32).cpu().detach().numpy())
                        image_mri.save(mri_str_path)
                        image_pet.save(pet_str_path)
    
    loop.set_postfix({'D_loss': '{0:1.8f}'.format((D_M_loss_total+D_P_loss_total).sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'D_M_loss': '{0:1.8f}'.format(D_M_loss_total.sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'D_P_loss': '{0:1.8f}'.format(D_P_loss_total.sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'G_Loss': '{0:1.8f}'.format(G_loss_total.sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'G_GAN_Loss': '{0:1.8f}'.format(G_GAN_Loss_total.sum().cpu().detach().numpy().item()/(batch_idx+1))})   

    return (D_P_loss_total.sum().cpu().detach().numpy().item()/(batch_idx+1)), \
           (D_M_loss_total.sum().cpu().detach().numpy().item()/(batch_idx+1)), \
           (G_loss_total.sum().cpu().detach().numpy().item()/(batch_idx+1))
        

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

    gpus = [0,1,3,4,5,6,7]

    G_M = generate_ConvTrantGe(args)
    G_P = generate_ConvTrantGe(args)
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

    L1 = nn.L1Loss()
    #criterionGAN = GANLoss('lsgan')
    criterionGAN = nn.MSELoss()
    fake_mri_pool =  ImagePool(args.batch_size)
    fake_pet_pool = ImagePool(args.batch_size)
    train_loader, _ = get_data_loader(args)
    #val_loader = None
    #train_loader = get_data_loader(args)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    G_loss_array=[]
    D_loss_mri_array=[]
    D_loss_pet_array=[]

    
    opt_gen = optim.Adam(
        list(G_M.parameters()) + list(G_P.parameters()),
        lr=args.lr,
         betas=(0.5, 0.999),
    )

    opt_disc = optim.Adam(
        list(D_M.parameters()) + list(D_P.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=100)
    scheduler_dis_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=100)

    for epoch in tqdm(range(args.max_iter)):
        # if epoch < 1e4:
        #     warmup_learning_rate(iteration_count=epoch, args=args, opt_disc=opt_disc, opt_gen=opt_gen)
        # else:
        #     adjust_learning_rate(iteration_count=epoch, args=args, opt_disc=opt_disc, opt_gen=opt_gen)
        print()
        print("Learning rate is {}".format(opt_gen.param_groups[0]['lr']))
        D_pet_loss, D_mri_loss, G_loss = train(train_loader, args, G_M, G_P, D_P, D_M, L1, 
                                               criterionGAN, opt_disc, opt_gen, g_scaler, d_scaler, device, writer, epoch, fake_mri_pool, fake_pet_pool)
        scheduler_gen.step()
        scheduler_dis_p.step()
        G_loss_array.append(G_loss)
        D_loss_mri_array.append(D_mri_loss)
        D_loss_pet_array.append(D_pet_loss)
    
    epoch_array = np.array(range(1, args.max_iter+1))
    G_loss_array = np.array(G_loss_array)
    D_loss_mri_array =np.array(D_loss_mri_array)
    D_loss_pet_array =np.array(D_loss_pet_array)
    show_loss_graph(epoch_array, G_loss_array, "./Generator_loss_mse.jpg", "Generator Loss")
    show_loss_graph(epoch_array, D_loss_mri_array, "./Disc_mri_loss_mse.jpg", "Discriminator MRI Loss")
    show_loss_graph(epoch_array, D_loss_pet_array, "./Disc_pet_loss.jpg_mse", "Discriminator PET Loss")
    
    # save_checkpoint(G_M, opt_gen, filename=args.checkpoint_gen_m)
    # save_checkpoint(G_P, opt_gen, filename=args.checkpoint_gen_p)
    # save_checkpoint(D_P, opt_disc, filename=args.checkpoint_disc_p)
    # save_checkpoint(D_M, opt_disc, filename=args.checkpoint_disc_m)

    save_checkpoint(G_M, opt_gen, filename='./saved/genm_new.pth.tar')
    save_checkpoint(G_P, opt_gen, filename='./saved/genp_new.pth.tar')
    save_checkpoint(D_P, opt_disc, filename='./saved/discm_new.pth.tar')
    save_checkpoint(D_M, opt_disc, filename='./saved/discp_new.pth.tar')



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
    parser.add_argument('--max_iter', type=int, default=1600)
    parser.add_argument('--batch_size', type=int, default=7) 
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
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--prin_frep', type=float, default=2, help='how much frequence we want to print')
    args = parser.parse_args()

    main(args)
