from ignite.metrics import PSNR
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from sklearn.metrics import mean_absolute_error
import torch
from torchmetrics import PearsonCorrCoef
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
import torchio as tio
import nibabel as nib


def eval_step(engine, batch):
    return batch


def Compute_PSNR(original, generated):
    # psnr = PSNR(data_range=1.0)
    # default_evaluator = Engine(eval_step)
    # psnr.attach(default_evaluator, 'psnr')
    # state = default_evaluator.run([[original, generated]])
    # return state.metrics['psnr']
    psnr = PeakSignalNoiseRatio()
    return psnr(generated, original)


def compute_SSIM(original, generated): 
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(generated, original)


def compute_CorCoef(original, generated):
    original = torch.flatten(original)
    generated = torch.flatten(generated)
    pearson = PearsonCorrCoef()
    return pearson(generated, original)


def compute_MAE(original, generated):
    original = torch.flatten(original)
    generated = torch.flatten(generated)
    return mean_absolute_error(original, generated)
   

if __name__ == '__main__':
    
    # img1 = torch.randn(1, 1, 48, 64, 48)
    # img2 = torch.randn(1, 1, 48, 64, 48)
    fake_mri_path = "/u/z/h/zhaoyang/CVLab/formal_project/train_data/pet/pet_train_batch2_data_1.nii.gz"
    fake_pet_path = "/u/z/h/zhaoyang/CVLab/formal_project/train_data/mri/mri_train_batch2_data_1.nii.gz"
    true_mri_path = "/u/z/h/zhaoyang/CVLab/formal_project/nii_data/train/mri/mri_train_8_data.nii.gz"
    true_pet_path = "/u/z/h/zhaoyang/CVLab/formal_project/nii_data/train/pet/pet_train_8_data.nii.gz"
    fake_mri = torch.from_numpy(nib.load(fake_mri_path).get_fdata())
    fake_pet = torch.from_numpy(nib.load(fake_pet_path).get_fdata())
    true_mri = torch.from_numpy(nib.load(true_mri_path).get_fdata())
    true_pet = torch.from_numpy(nib.load(true_pet_path).get_fdata())
    corCoef = compute_CorCoef(true_pet, fake_pet)
    mae = compute_MAE(true_pet, fake_pet)
    fake_mri = fake_mri.unsqueeze(dim=0)
    fake_pet = fake_pet.unsqueeze(dim=0)
    true_mri = true_mri.unsqueeze(dim=0)
    true_pet = true_pet.unsqueeze(dim=0)
    psnr = Compute_PSNR(true_pet, fake_pet)
    ssim = compute_SSIM(true_pet, fake_pet)
    print("corCoef is:{}".format(corCoef))
    print("MAE is:{}".format(mae))
    print("PSNR is:{}".format(psnr))
    print("SSIM is:{}".format(ssim))

    