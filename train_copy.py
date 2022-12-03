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
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from Brain_img.read_3D import get_train_loader
import itertools
from utils.utils import LambdaLR
from torch.autograd import Variable


GRADIENT_ACCUMULATION = 2


def show_loss_graph(epoch_array, G_loss_array, D_pet_array, D_mri_array):
    plt.figure(figsize=(12,8))
    plt.plot(epoch_array, G_loss_array, marker='*', label='G_loss', color='r')
    plt.plot(epoch_array, D_pet_array, marker='o', label='D_pet_loss', color='b')
    plt.plot(epoch_array, D_mri_array, marker='.', label='D_mri_loss', color='g')
    plt.xlabel("Epoch")
    plt.xlabel("Loss")
    plt.legend()
    plt.savefig("loss.jpg")
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


def adjust_learning_rate(optimizers_model, iteration_count, args):
    """Imitating the original implementation"""
    args.lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for _, optimizer in optimizers_model.items():
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    


def warmup_learning_rate(optimizers_model, iteration_count, args):
    """Imitating the original implementation"""
    args.lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for _, optimizer in optimizers_model.items():
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    


#def train(train_loader, args, G_M, G_P, D_P, D_M, optimizers_model, Losses_Dic, device, writer, epoch, target_real, target_fake):
def train(train_loader, args, G_M, G_P, D_P, D_M, optimizers_model, Losses_Dic, device, writer, epoch):
    loop = tqdm(train_loader, leave=True)
    loop.set_postfix
    #print(out.shape)
    USE_CUDA = torch.cuda.is_available()
    #device = torch.device("cuda:0" if USE_CUDA else "cpu")
    #batch_idx = 0

    for batch_idx, batch in enumerate(train_loader):
        #loop.update(1)
        img_mri = batch['img_mri'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3).to(device)
        img_pet = batch['img_pet'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3).to(device)
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        img_mri = F.interpolate(img_mri, size=(96, 128, 96), mode='nearest').to(device)
        img_pet = F.interpolate(img_pet, size=(96, 128, 96), mode='nearest').to(device)

        G_loss_sum = 0
        D_mri_loss_sum = 0
        D_pet_loss_sum = 0
        ###### Gneerators mri2pet and pet2mri ######
        optimizers_model['optimizer_G'].zero_grad()
        with torch.cuda.amp.autocast():
            # Identity loss: mri
            same_mri = G_M(img_pet)
            loss_indentity_mri= Losses_Dic['identity'](same_mri, img_mri) * args.lamda_identity
            # Identity loss: pet
            same_pet = G_P(img_mri)
            loss_indentity_pet = Losses_Dic['identity'](same_pet, img_pet) * args.lamda_identity
            del same_mri, same_pet

            # GAN loss
            fake_mri = G_M(img_pet)
            pred_fake = D_M(fake_mri)
            #loss_GAN_pet2mri = Losses_Dic['GAN'](pred_fake, target_real)
            loss_GAN_pet2mri = Losses_Dic['GAN'](pred_fake, torch.ones_like(pred_fake))

            fake_pet = G_P(img_mri)
            pred_fake = D_P(fake_pet)
            #Lloss_GAN_mri2pet = Losses_Dic['GAN'](pred_fake, target_real)
            loss_GAN_mri2pet = Losses_Dic['GAN'](pred_fake, torch.ones_like(pred_fake))

            # Cycle loss
            recoverd_mri = G_M(fake_pet)
            loss_cycle_pet2mri = Losses_Dic['cycle'](recoverd_mri, img_mri) * args.lamda_cycle
            del recoverd_mri

            revocerd_pet = G_P(fake_mri)
            loss_cycle_mri2pet = Losses_Dic['cycle'](revocerd_pet, img_pet) * args.lamda_cycle
            del revocerd_pet

            # Total loss
            loss_G = loss_indentity_mri + loss_indentity_pet + \
                     loss_GAN_pet2mri + loss_GAN_mri2pet + \
                     loss_cycle_pet2mri + loss_cycle_mri2pet
            G_loss_sum += loss_G.item()
        
        loss_G.backward()
        if ((batch_idx + 1) % GRADIENT_ACCUMULATION == 0) or (batch_idx + 1 ==  len(list(train_loader))):
            optimizers_model['optimizer_G'].step()

        ###### Distriminator PET ######
        optimizers_model['optimizer_D_P'].zero_grad()
         
        with torch.cuda.amp.autocast():
            # Real loss
            pred_real = D_P(img_pet)
            # loss_pet_real = Losses_Dic['GAN'](pred_real, target_real) 
            loss_pet_real = Losses_Dic['GAN'](pred_real, torch.ones_like(pred_real)) 
            # Fake loss
            pred_fake = D_P(fake_pet.detach())
            # loss_pet_fake = Losses_Dic['GAN'](pred_fake, target_fake)
            loss_pet_fake = Losses_Dic['GAN'](pred_fake, torch.zeros_like(pred_fake))
            # Total loss
            loss_D_pet = (loss_pet_real + loss_pet_fake)*0.5
            D_pet_loss_sum += loss_D_pet.item()


        loss_D_pet.backward()
        if ((batch_idx + 1) % GRADIENT_ACCUMULATION == 0) or (batch_idx + 1 ==  len(list(train_loader))):
            optimizers_model['optimizer_D_P'].step()

        ###### Discriminator MRI ######
        optimizers_model['optimizer_D_M'].zero_grad()

        with torch.cuda.amp.autocast():
            # Real loss
            pred_real = D_M(img_mri)
            # loss_mri_real = Losses_Dic['GAN'](pred_real, target_real)
            loss_mri_real = Losses_Dic['GAN'](pred_real, torch.ones_like(pred_real))
            # Fake loss
            pred_fake = D_M(fake_mri.detach())
            loss_mri_fake = Losses_Dic['GAN'](pred_fake, torch.zeros_like(pred_fake))
            # Total loss
            loss_D_mri =  (loss_mri_real + loss_mri_fake)*0.5
            D_mri_loss_sum += loss_D_mri.item()

        
        loss_D_mri.backward()

        if ((batch_idx + 1) % GRADIENT_ACCUMULATION == 0) or (batch_idx + 1 ==  len(list(train_loader))):
            optimizers_model['optimizer_D_M'].step()

        ###### Save image of generating ######
        if batch_idx % 1 == 0:
            mri_str_path = ("./train_data" + "/mri_train_{}_data.nii.gz").format(batch_idx+1)
            pet_str_path = ("./train_data" + "/pet_train_{}_data.nii.gz").format(batch_idx+1)
            fake_mri = F.interpolate(fake_mri, size=(48, 64, 48), mode='nearest')
            fake_pet = F.interpolate(fake_pet, size=(48, 64, 48), mode='nearest')
            image_mri = tio.ScalarImage(tensor=fake_mri[0, :, :, :, :].type(torch.float32).cpu().detach().numpy())
            image_pet = tio.ScalarImage(tensor=fake_pet[0, :, :, :, :].type(torch.float32).cpu().detach().numpy())
            image_mri.save(mri_str_path)
            image_pet.save(pet_str_path)

        # loop.set_postfix({'pet_loss': '{0:1.7f}'.format(loss_D_pet.sum().cpu().detach().numpy().item()),
        #                   'pet_fake_loss': '{0:1.7f}'.format(loss_pet_fake.sum().cpu().detach().numpy().item()),
        #                   'pet_real_loss': '{0:1.7f}'.format(loss_pet_real.sum().cpu().detach().numpy().item()),
        #                   'mri_loss': '{0:1.7f}'.format(loss_D_mri.sum().cpu().detach().numpy().item()),
        #                   'mri_fake_loss': '{0:1.7f}'.format(loss_mri_fake.sum().cpu().detach().numpy().item()),
        #                   'mri_real_loss': '{0:1.7f}'.format(loss_mri_real.sum().cpu().detach().numpy().item()),
        #                   'G_Loss': '{0:1.7f}'.format(loss_G.sum().cpu().detach().numpy().item()),
        #                   'loss_G_identity': '{0:1.7f}'.format((loss_indentity_mri + loss_indentity_pet).sum().cpu().detach().numpy().item()),
        #                   'loss_G_GAN': '{0:1.7f}'.format((loss_GAN_pet2mri + Lloss_GAN_mri2pet).sum().cpu().detach().numpy().item()),
        #                   'loss_G_cycle': '{0:1.7f}'.format((loss_cycle_pet2mri + loss_cycle_mri2pet).sum().cpu().detach().numpy().item()) })
    
    loop.set_postfix({'avg_D_mri_loss': '{0:1.5f}'.format(D_mri_loss_sum/(batch_idx+1)),
                      'avg_D_pet_loss': '{0:1.5f}'.format(D_pet_loss_sum/(batch_idx+1)),
                      'avg_G_Loss': '{0:1.5f}'.format(G_loss_sum/(batch_idx+1))})

    return G_loss_sum/(batch_idx+1), \
           D_pet_loss_sum/(batch_idx+1), \
           D_mri_loss_sum/(batch_idx+1)
    



def main(args):

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    G_M = generate_ConvTrantGe(args)
    G_P = generate_ConvTrantGe(args)
    D_M = generate_dis(args)
    D_P = generate_dis(args)

    G_M = nn.DataParallel(G_M)
    G_P = nn.DataParallel(G_P)
    D_M = nn.DataParallel(D_M)
    D_P = nn.DataParallel(D_P)

    G_M.to(device)
    G_P.to(device)
    D_M.to(device)
    D_P.to(device)
  
    # Lossess
    Losses_Dic = {'GAN':torch.nn.MSELoss(), 'cycle': torch.nn.L1Loss(), 'identity':torch.nn.L1Loss()}

    # Optimizers & LR schedulers
    optimizers_model = {'optimizer_G':torch.optim.Adam(itertools.chain(G_M.parameters(), G_P.parameters()),lr=args.lr, betas=(0.5, 0.999)),
                        'optimizer_D_P': torch.optim.Adam(D_P.parameters(), lr=args.lr, betas=(0.5, 0.999)),
                        'optimizer_D_M': torch.optim.Adam(D_M.parameters(), lr=args.lr, betas=(0.5, 0.999))}

    
    train_loader = get_train_loader(args)
    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
    #target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False).to(device)
    #target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False).to(device)
    

    #fake_mri_buffer = ReplayBuffer()
    #fake_pet_buffer = ReplayBuffer()


    G_loss_array=[]
    D_pet_array=[]
    D_mri_array=[]

    for epoch in tqdm(range(args.n_epochs)):
        if epoch < 1e4:
            warmup_learning_rate(optimizers_model, iteration_count=epoch, args=args)
        else:
            adjust_learning_rate(optimizers_model, iteration_count=epoch, args=args)
        print("Learning rate is {}".format(args.lr))
        #G_loss, D_pet_loss, D_mri_loss = train(train_loader, args, G_M, G_P, D_P, D_M, optimizers_model, Losses_Dic, device, writer, epoch, target_real, target_fake)
        G_loss, D_pet_loss, D_mri_loss = train(train_loader, args, G_M, G_P, D_P, D_M, optimizers_model, Losses_Dic, device, writer, epoch)
        G_loss_array.append(G_loss)
        D_pet_array.append(D_pet_loss)
        D_mri_array.append(D_mri_loss)
    
        torch.save(G_M.state_dict(), args.checkpoint_gen_m)
        torch.save(G_P.state_dict(), args.checkpoint_gen_p)
        torch.save(D_M.state_dict(), args.checkpoint_disc_m)
        torch.save(D_P.state_dict(), args.checkpoint_disc_p)
    
    epoch_array = np.array(range(1, args.n_epochs+1))
    G_loss_array = np.array(G_loss_array)
    D_pet_array = np.array(D_pet_array)
    D_mri_array = np.array(D_mri_array)
    show_loss_graph(epoch_array, G_loss_array, D_pet_array, D_mri_array)
    
    




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
    parser.add_argument('--input_c', default=1, type=int)
    parser.add_argument('--input_w', default=96, type=int)
    parser.add_argument('--input_h', default=128, type=int)
    parser.add_argument('--input_d', default=96, type=int)
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--cuda', default=True , help='if true, cuda is to be used')

    # train options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--lamda_cycle', type=float, default=10)
    parser.add_argument('--lamda_identity', type=float, default=0.5)
    parser.add_argument('--checkpoint_gen_m', default='./saved/genm.pth.tar')
    parser.add_argument('--checkpoint_gen_p', default='./saved/genp.pth.tar')
    parser.add_argument('--checkpoint_disc_m', default='./saved/discm.pth.tar')
    parser.add_argument('--checkpoint_disc_p', default='./saved/discp.pth.tar')
    args = parser.parse_args()

    main(args)
