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
#from Brain_img_V2.get_data_loader import get_data_loader

GRADIENT_ACCUMULATION = 3


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


def adjust_learning_rate(iteration_count, args):
    """Imitating the original implementation"""
    args.lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    


def warmup_learning_rate(iteration_count, args):
    """Imitating the original implementation"""
    args.lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    


def train(train_loader, val_loader, args, G_M, G_P, D_P, D_M, L1, MSE, opt_disc, opt_gen, g_scaler, d_scaler, device, writer, epoch):
    loop = tqdm(train_loader, leave=True)
    loop.set_postfix
    #print(out.shape)
    USE_CUDA = torch.cuda.is_available()
    #device = torch.device("cuda:0" if USE_CUDA else "cpu")
    #batch_idx = 0
    G_loss=None
    D_loss=None

    batch_idx=0
    for batch_idx, batch in enumerate(train_loader):
    #for img_mri, img_pet in train_loader:
        img_mri = batch['img_mri'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3)
        img_pet = batch['img_pet'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3)
        #img_mri = img_mri.permute(0, 4, 1, 2, 3)
        #img_pet = img_pet.permute(0, 4, 1, 2, 3)
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        #N, C, D, H, W = content_images.shape
        img_mri = F.interpolate(img_mri, size=(96, 128, 96), mode='nearest').to(device)
        img_pet = F.interpolate(img_pet, size=(96, 128, 96), mode='nearest').to(device)
        #mri_pet = torch.stack((image_mri, img_pet), 0)
        #del img_mri, img_pet
        
        with torch.cuda.amp.autocast():
            fake_pet = G_P(img_mri)
            D_P_real = D_P(img_pet)
            D_P_fake = D_P(fake_pet.detach())
            D_P_real_loss = MSE(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = MSE(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            fake_mri = G_M(img_pet)
            D_M_real = D_M(img_mri)
            D_M_fake = D_M(fake_mri.detach())
            D_M_real_loss = MSE(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = MSE(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            D_loss = (D_P_loss + D_M_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss / GRADIENT_ACCUMULATION).backward()
        if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0 or (batch_idx + 1) ==  len(list(train_loader)) == 0:
                d_scaler.step(opt_disc)
                d_scaler.update()
        # d_scaler.scale(D_loss).backward()
        # d_scaler.step(opt_disc)
        # d_scaler.update()
        
        # Train Generator
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_P_fake = D_P(fake_pet)
            D_M_fake = D_M(fake_mri)
            loss_G_M = MSE(D_M_fake, torch.ones_like(D_M_fake))
            loss_G_P = MSE(D_P_fake, torch.ones_like(D_P_fake))

            # cycle loss
            cycle_mri = G_M(fake_pet)
            cycle_mri_loss = L1(img_mri, cycle_mri)
            del cycle_mri

            cycle_pet = G_P(fake_mri)
            cycle_pet_loss = L1(img_pet, cycle_pet)
            del cycle_pet

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_mri = G_M(img_mri)
            identity_mri_loss = L1(img_mri, identity_mri)
            del identity_mri

            identity_pet = G_P(img_pet)
            identity_pet_loss = L1(img_pet, identity_pet)
            del identity_pet

            G_loss = (   # six loss 
                loss_G_P
                + loss_G_M
                + cycle_mri_loss * args.lamda_cycle
                + cycle_pet_loss *  args.lamda_cycle
                + identity_mri_loss * args.lamda_identity
                + identity_pet_loss * args.lamda_identity
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss / GRADIENT_ACCUMULATION).backward()
        if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0 or (batch_idx + 1) ==  len(list(train_loader)):
                g_scaler.step(opt_gen)
                g_scaler.update()
        # g_scaler.scale(G_loss).backward()
        # g_scaler.step(opt_gen)
        # g_scaler.update()
        

        writer.add_scalar('Generator_Loss', G_loss.sum().item(), epoch + 1, batch_idx+1)
        writer.add_scalar('Discriminator_Loss', D_loss.sum().item(), epoch + 1, batch_idx+1)
        #batch_idx = batch_idx + 1

        if batch_idx % 1 == 0:
            mri_str_path = ("./train_data" + "/mri_train_{}_data.nii.gz").format(batch_idx+1)
            pet_str_path = ("./train_data" + "/pet_train_{}_data.nii.gz").format(batch_idx+1)
            fake_mri = F.interpolate(fake_mri, size=(48, 64, 48), mode='nearest')
            fake_pet = F.interpolate(fake_pet, size=(48, 64, 48), mode='nearest')
            image_mri = tio.ScalarImage(tensor=fake_mri[0, :, :, :, :].type(torch.float32).cpu().detach().numpy())
            image_pet = tio.ScalarImage(tensor=fake_pet[0, :, :, :, :].type(torch.float32).cpu().detach().numpy())
            image_mri.save(mri_str_path)
            image_pet.save(pet_str_path)
    
    loop.set_postfix({'D_loss': '{0:1.8f}'.format(D_loss.sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'pet_real_loss': '{0:1.8f}'.format(D_P_real_loss.sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'pet_fake_loss': '{0:1.8f}'.format(D_P_fake_loss.sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'mri_real_loss': '{0:1.8f}'.format(D_M_real_loss.sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'mri_fake_loss': '{0:1.8f}'.format(D_M_fake_loss.sum().cpu().detach().numpy().item()/(batch_idx+1)),
                      'G_Loss': '{0:1.8f}'.format(G_loss.sum().cpu().detach().numpy().item()/(batch_idx+1))})

    return (D_P_loss.sum().cpu().detach().numpy().item()/(batch_idx+1)), \
           (D_M_loss.sum().cpu().detach().numpy().item()/(batch_idx+1)), \
           (G_loss.sum().cpu().detach().numpy().item()/(batch_idx+1))
        

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

    gpus = [0,1,2,3]

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
  
    opt_disc = optim.Adam(
        list(D_M.parameters()) + list(D_P.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    # opt_dism = optim.Adam(D_M.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # opt_disp = optim.Adam(D_P.parameters(), lr=args.lr, betas=(0.5, 0.999))

    opt_gen = optim.Adam(
        list(G_M.parameters()) + list(G_P.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    train_loader, val_loader = get_data_loader(args)
    #val_loader = None
    #train_loader = get_data_loader(args)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    G_loss_array=[]
    D_loss_mri_array=[]
    D_loss_pet_array=[]

    for epoch in tqdm(range(args.max_iter)):
        if epoch < 1e4:
            warmup_learning_rate(iteration_count=epoch, args=args)
        else:
            adjust_learning_rate(iteration_count=epoch, args=args)
        print("Learning rate is {}".format(args.lr))
        D_pet_loss, D_mri_loss, G_loss = train(train_loader, val_loader, args, G_M, G_P, D_P, D_M, L1, MSE, opt_disc, opt_gen, g_scaler, d_scaler, device, writer, epoch)
        G_loss_array.append(G_loss)
        D_loss_mri_array.append(D_mri_loss)
        D_loss_pet_array.append(D_pet_loss)
    
    epoch_array = np.array(range(1, args.max_iter+1))
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
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4) 
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=7.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=300)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--lamda_cycle', type=float, default=10)
    parser.add_argument('--lamda_identity', type=float, default=0.5)
    parser.add_argument('--checkpoint_gen_m', default='./saved/genm.pth.tar')
    parser.add_argument('--checkpoint_gen_p', default='./saved/genp.pth.tar')
    parser.add_argument('--checkpoint_disc_m', default='./saved/discm.pth.tar')
    parser.add_argument('--checkpoint_disc_p', default='./saved/discp.pth.tar')
    args = parser.parse_args()

    main(args)
