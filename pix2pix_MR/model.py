import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image, make_grid
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
import random
import os
import itertools
from tqdm import tqdm
import h5py
import numpy as np

random.seed(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
EPOCHS = 100
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_l1 = 100
LAMBDA_perceptual = 100

class MRDataset(Dataset):
    def __init__(self, pdw_dir, t2w_dir, transform=None):
        # PDw와 T2w 이미지 파일 경로 목록을 불러옴
        self.pdw_files = sorted([os.path.join(pdw_dir, f) for f in os.listdir(pdw_dir) if f.endswith('.hdf5')])  # .hdf5 파일만 선택
        self.t2w_files = sorted([os.path.join(t2w_dir, f) for f in os.listdir(t2w_dir) if f.endswith('.hdf5')])  # .hdf5 파일만 선택
        # random.shuffle(self.t2w_files)
        
        self.transform = transform

    def __getitem__(self, index):
        # PDw와 T2w 이미지 로드 (HDF5 파일에서 읽음)
        pdw_file = self.pdw_files[index]
        t2w_file = self.t2w_files[index]
        
        # HDF5 파일 열기
        with h5py.File(pdw_file, 'r') as pdw_data, h5py.File(t2w_file, 'r') as t2w_data:
            pdw_img = pdw_data['array'][:][0]  # 'array' 키에 해당하는 데이터를 읽음
            t2w_img = t2w_data['array'][:][0]  # 'array' 키에 해당하는 데이터를 읽음

        pdw_img = ((pdw_img - pdw_img.min()) / (pdw_img.max() - pdw_img.min()) * 255).astype(np.uint8)
        t2w_img = ((t2w_img - t2w_img.min()) / (t2w_img.max() - t2w_img.min()) * 255).astype(np.uint8)
        
        pdw_img = Image.fromarray(pdw_img, mode='L') # 이미지로 변환
        t2w_img = Image.fromarray(t2w_img, mode='L') # 이미지로 변환
        
        # 이미지에 변환 적용
        pdw_img = self.transform(pdw_img)
        t2w_img = self.transform(t2w_img)
        
        return {'PDw': pdw_img, 'T2w': t2w_img}

    def __len__(self):
        return len(self.pdw_files)  # 두 디렉토리 내 파일 수의 최소값을 반환

# 데이터 변환 및 로더 설정
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])


dataset = MRDataset(
    pdw_dir='/mnt/MW/data/IXI datasets/h5/train/PDw',
    t2w_dir='/mnt/MW/data/IXI datasets/h5/train/T2w',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, in_channels = 1, num_residual_blocks = 9):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]


        in_features = 64
        out_features = 2 * in_features
        for _ in range(2):
            model += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
        
            in_features = out_features
            out_features = 2 * in_features

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features//2

        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.Conv2d(64, 1, kernel_size=7, padding = 3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks).to(device)
        self.transform = nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to(device)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to(device)
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += self.criterion(x, y)
        return loss

# 가중치 초기화 함수
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

G_PDw2T2w = Generator().to(device).apply(weights_init_normal)
D_T2w = Discriminator().to(device).apply(weights_init_normal)

criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()
criterion_Perceptual = VGGPerceptualLoss()

# 옵티마이저
optimizer_G = optim.Adam(
    itertools.chain(G_PDw2T2w.parameters()),
    lr=LR, betas=(BETA1, BETA2)
)
optimizer_D = optim.Adam(
    itertools.chain(D_T2w.parameters()),
    lr=LR, betas=(BETA1, BETA2)
)

# Learning rate schedulers
def lambda_rule(epoch):
    lr_l = 1.0
    if epoch > 50:
        lr_l = 1.0 - max(0, epoch - 50) / float(50)
    return lr_l

scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

def save_checkpoint(epoch, G_PDw2T2w, D_T2w,
                   optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    """모델 체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'G_PDw2T2w_state_dict': G_PDw2T2w.state_dict(),
        'D_T2w_state_dict': D_T2w.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
    }
    save_path = os.path.join('/mnt/MW/checkpoints/pix2pix_MR/', f'checkpoint_latest.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: checkpoint_latest.pth")

def load_checkpoint(checkpoint_path, G_PDw2T2w, D_T2w,
                   optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    """모델 체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path)

    G_PDw2T2w.load_state_dict(checkpoint['G_PDw2T2w_state_dict'])
    D_T2w.load_state_dict(checkpoint['D_T2w_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
    scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch

def train():
    torch.backends.cudnn.benchmark = True
    start_epoch = 0

    # 체크포인트가 있다면 로드
    if os.path.exists('/mnt/MW/checkpoints/pix2pix_MR/checkpoint_latest.pth'):
        start_epoch = load_checkpoint('/mnt/MW/checkpoints/pix2pix_MR/checkpoint_latest.pth',
                                    G_PDw2T2w, D_T2w,
                                    optimizer_G, optimizer_D, scheduler_G, scheduler_D)
        start_epoch += 1  # 다음 에포크부터 시작

    for epoch in range(start_epoch, EPOCHS+1):
        # tqdm을 이용해 에포크 진행상황 표시
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{EPOCHS}") as pbar:
            for i, batch in pbar:
                real_PDw = batch['PDw'].to(device)
                real_T2w = batch['T2w'].to(device)
                

                # 진짜 및 가짜 레이블 - adjust size for single cross-section
                valid = torch.ones((real_PDw.size(0), 1, 16, 16)).to(device)
                fake = torch.zeros((real_PDw.size(0), 1, 16, 16)).to(device)

                optimizer_G.zero_grad()

                # GAN loss
                fake_T2w = G_PDw2T2w(real_PDw)

                loss_GAN_real = criterion_GAN(D_T2w(real_T2w), valid)  # 진짜 이미지에 대한 판별자 손실
                loss_GAN_fake = criterion_GAN(D_T2w(fake_T2w.detach()), fake)  # 가짜 이미지에 대한 판별자 손실
                loss_GAN = - (loss_GAN_real + loss_GAN_fake)
                # L1 loss
                loss_l1 = criterion_L1(fake_T2w, real_T2w)

                # Perceptual loss
                perceptual_loss = criterion_Perceptual(fake_T2w, real_T2w)

                # Total loss
                loss_G = loss_GAN + LAMBDA_l1 * loss_l1 + LAMBDA_perceptual * perceptual_loss

                loss_G.backward()
                optimizer_G.step()

                # Discriminator 학습 
                optimizer_D.zero_grad()


                loss_real_T2w = criterion_GAN(D_T2w(real_T2w), valid)
                loss_fake_T2w = criterion_GAN(D_T2w(fake_T2w.detach()), fake)
                loss_D_T2w = (loss_real_T2w + loss_fake_T2w) / 4
                loss_D_T2w.backward()

                optimizer_D.step()

                if i % 100 == 0:
                    pbar.set_postfix({
                        'D_loss': loss_D_T2w.item(),
                        'G_loss': loss_G.item(),
                        'Perceptual': perceptual_loss.item()
                    })

        scheduler_G.step()
        scheduler_D.step()

        # 체크포인트 저장
        save_checkpoint(epoch, G_PDw2T2w, D_T2w, optimizer_G, optimizer_D, scheduler_G, scheduler_D)