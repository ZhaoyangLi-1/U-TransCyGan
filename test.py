import torch
from models.generator import generate_ConvTrantGe
from models.discriminator import generate_dis 
from Brain_img.read_3D import get_val_loader
import argparse
import torch.optim as optim
import torch.nn.functional as F
import os
import torchio as tio
from torch import nn
#hello

def load_model(model, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['state_dict'])
    


def save_image(file_path, img, type, ith_image, real):
    batch_size, _, _, _, _ = img.size()
    print("{} Image".format(type))
    for i in range(batch_size):
        print("Saving {} {} image".format(ith_image, real))
        slice_img =  tio.ScalarImage(tensor=img[i, :, :, :, :].cpu().detach())
        saved_file_path = file_path + "/" + type + "_{}.nii.gz".format(ith_image)
        slice_img.save(saved_file_path)
        ith_image += 1



def main(args):
    G_M = generate_ConvTrantGe(args)
    G_P = generate_ConvTrantGe(args)
    

    opt_gen = optim.Adam(
        list(G_M.parameters()) + list(G_P.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )


    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:4" if USE_CUDA else "cpu")

    gpus = [4,5,6]

    G_M = nn.DataParallel(G_M, device_ids=gpus, output_device=gpus[0])
    G_P = nn.DataParallel(G_P, device_ids=gpus, output_device=gpus[0])

    G_M.to(device)
    G_P.to(device)

    load_model(G_P, args.checkpoint_gen_p)
    load_model(G_M, args.checkpoint_gen_m)

    G_M.eval()
    G_P.eval()


    val_loader = get_val_loader(args)
    ith_image_mri = 0
    ith_image_pet = 0
    for idx, batch in enumerate(val_loader):
        if idx is 3: break
        img_mri = batch['img_mri'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3)
        img_pet = batch['img_pet'].reshape((args.batch_size, 48, 64, 48, 1)).permute(0, 4, 1, 2, 3)
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        #N, C, D, H, W = content_images.shape
        img_mri = F.interpolate(img_mri, size=(96, 128, 96), mode='nearest').to(device)
        img_pet = F.interpolate(img_pet, size=(96, 128, 96), mode='nearest').to(device)
        with torch.no_grad():
            out_pet = G_P(img_mri)
            out_mri = G_M(img_pet)
        
        out_pet = F.interpolate(out_pet, size=(48, 64, 48), mode='nearest')
        #out_pet = out_pet.type(torch.float32).cpu().detach().numpy()
        out_pet = out_pet.type(torch.float32)

        out_mri = F.interpolate(out_mri, size=(48, 64, 48), mode='nearest')
        #out_mri = out_mri.type(torch.float32).cpu().detach().numpy()
        out_mri = out_mri.type(torch.float32)
        
        save_image(args.test_save_mri, out_mri, "mri", ith_image_mri, "output")
        save_image(args.test_save_pet, out_pet, "pet", ith_image_pet, "output")
        ith_image_mri = 0 
        ith_image_pet = 0
        print("Saving Changed Test Data")
        save_image("./nii_data/changed_val/mri", img_mri, "mri", ith_image_mri, "real")
        save_image("./nii_data/changed_val/pet", img_pet, "pet", ith_image_pet, "real")
        break

        

        



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

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--input_w', default=96, type=int)
    parser.add_argument('--input_h', default=128, type=int)
    parser.add_argument('--input_d', default=96, type=int)
    parser.add_argument('--batch_size', type=int, default=5) 
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=7.0)
    parser.add_argument('--n_thr    eads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
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
    parser.add_argument('--test_save_mri', default='./test_save/mri')
    parser.add_argument('--test_save_pet', default='./test_save/pet')
    args = parser.parse_args()
    
    main(args)