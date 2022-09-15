from models.generator import generate_ConvTrantGe
from models.discriminator import generate_dis
import torch
import torch.nn as nn

class U_CTransGan(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.G_M = generate_ConvTrantGe(args)
        self.G_P = generate_ConvTrantGe(args)
        self.D_M = generate_dis(args)
        self.D_P = generate_dis(args)
        self.lamda_cycle = args.lamda_cycle
        self.lamda_identity = args.lamda_identity
        self.train_dis = False
        self.train_gen = False

        self.L1 = nn.L1Loss()
        self.MSE = nn.MSELoss()
    
    def forward(self, x):
        img_mri = x[0]
        img_pet =  x[1]

        if self.train_dis is True and self.train_gen is False:
            fake_pet = self.G_P(img_mri)
            D_P_real = self.D_P(img_pet)
            D_P_fake = self.D_P(fake_pet.detach())
            D_P_real_loss = self.MSE(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = self.MSE(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss
            del D_P_real, D_P_fake, D_P_real_loss, D_P_fake_loss

            fake_mri = self.G_M(img_pet)
            D_M_real = self.D_M(img_mri)
            D_M_fake = self.D_M(fake_mri.detach())
            D_M_real_loss = self.MSE(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = self.MSE(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss
            del D_M_real, D_M_fake, D_M_real_loss, D_M_fake_loss

            D_loss = (D_P_loss + D_M_loss) / 2
            
            return torch.stack((fake_mri, fake_pet), 0), D_loss
        
        if self.train_dis is False and self.train_gen is True:
            # adversarial loss for both generators
            D_P_fake = self.D_P(fake_pet)
            loss_G_P = self.MSE(D_P_fake, torch.ones_like(D_P_fake))
            del D_P_fake
            D_M_fake = self.D_M(fake_mri)
            loss_G_M = self.MSE(D_M_fake, torch.ones_like(D_M_fake))
            del D_M_fake
           
            # cycle loss
            cycle_mri = self.G_M(fake_pet)
            cycle_mri_loss = self.L1(img_mri, cycle_mri)
            del cycle_mri

            cycle_pet = self.G_P(fake_mri)
            cycle_pet_loss = self.L1(img_pet, cycle_pet)
            del cycle_pet

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_mri = self.G_M(img_mri)
            identity_pet = self.G_P(img_pet)
            identity_mri_loss = self.L1(img_mri, identity_mri)
            identity_pet_loss = self.L1(img_pet, identity_pet)
            del identity_mri, identity_pet

            G_loss = (   # six loss 
                loss_G_P
                + loss_G_M
                + cycle_mri_loss * self.lamda_cycle
                + cycle_pet_loss *  self.lamda_cycle
                + identity_mri_loss * self.lamda_identity
                + identity_pet_loss * self.lamda_identity
            )

            return G_loss
