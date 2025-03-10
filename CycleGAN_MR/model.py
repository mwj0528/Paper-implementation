import torch
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

random.seed(42)
torch.manual_seed(42)

# 하이퍼파라미터 설정
BATCH_SIZE = 1
EPOCHS = 200
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_CYCLE = 100.0
# LAMBDA_IDENTITY = 0.5


class MRDataset(Dataset):
    def __init__(self, pdw_dir, t2w_dir, transform=None):
        # PDw와 T2w 이미지 파일 경로 목록을 불러옴
        self.pdw_files = [os.path.join(pdw_dir, f) for f in os.listdir(pdw_dir) if f.endswith('.png')]  # .png 파일만 선택
        self.t2w_files = [os.path.join(t2w_dir, f) for f in os.listdir(t2w_dir) if f.endswith('.png')]  # .png 파일만 선택
        random.shuffle(self.pdw_files)
        self.transform = transform 

    def __getitem__(self, index):
        # PDw와 T2w 이미지 로드
        pdw_img = Image.open(self.pdw_files[index]).convert('RGB')
        t2w_img = Image.open(self.t2w_files[index]).convert('RGB')

        # 이미지에 변환 적용
        pdw_img = self.transform(pdw_img)
        t2w_img = self.transform(t2w_img)

        return {'PDw': pdw_img, 'T2w': t2w_img}

    def __len__(self):
        return min(len(self.pdw_files), len(self.t2w_files))  # 두 디렉토리 내 파일 수의 최소값을 반환


# 데이터 변환 및 로더 설정
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MRDataset(
    pdw_dir='/mnt/MW/data/IXI datasets/png/train/PDw',
    t2w_dir='/mnt/MW/data/IXI datasets/png/train/T2w',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels = 3, num_residual_blocks = 9):
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
            nn.Conv2d(64, 3, kernel_size=7, padding = 3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_PDw2T2w = Generator().to(device).apply(weights_init_normal)
G_T2w2PDw = Generator().to(device).apply(weights_init_normal)
D_PDw = Discriminator().to(device).apply(weights_init_normal)
D_T2w = Discriminator().to(device).apply(weights_init_normal)

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# 옵티마이저
optimizer_G = optim.Adam(
    itertools.chain(G_PDw2T2w.parameters(), G_T2w2PDw.parameters()),
    lr=LR, betas=(BETA1, BETA2)
)
optimizer_D = optim.Adam(
    itertools.chain(D_PDw.parameters(), D_T2w.parameters()),
    lr=LR, betas=(BETA1, BETA2)
)

# Learning rate schedulers
def lambda_rule(epoch):
    lr_l = 1.0
    if epoch > 100:
        lr_l = 1.0 - max(0, epoch - 100) / float(100)
    return lr_l

scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

def save_checkpoint(epoch, G_PDw2T2w, G_T2w2PDw, D_PDw, D_T2w,
                   optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    """모델 체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'G_PDw2T2w_state_dict': G_PDw2T2w.state_dict(),
        'G_T2w2PDw_state_dict': G_T2w2PDw.state_dict(),
        'D_PDw_state_dict': D_PDw.state_dict(),
        'D_T2w_state_dict': D_T2w.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
    }
    save_path = os.path.join('/mnt/MW/checkpoints/CycleGAN/', f'checkpoint_latest.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: checkpoint_latest.pth")

def load_checkpoint(checkpoint_path, G_PDw2T2w, G_T2w2PDw, D_PDw, D_T2w,
                   optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    """모델 체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path)

    G_PDw2T2w.load_state_dict(checkpoint['G_PDw2T2w_state_dict'])
    G_T2w2PDw.load_state_dict(checkpoint['G_T2w2PDw_state_dict'])
    D_PDw.load_state_dict(checkpoint['D_PDw_state_dict'])
    D_T2w.load_state_dict(checkpoint['D_T2w_state_dict'])

    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
    scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch

def train():
    start_epoch = 0

    # 체크포인트가 있다면 로드
    if os.path.exists('/mnt/MW/checkpoints/CycleGAN/checkpoint_latest.pth'):
        start_epoch = load_checkpoint('/mnt/MW/checkpoints/CycleGAN/checkpoint_latest.pth',
                                    G_PDw2T2w, G_T2w2PDw, D_PDw, D_T2w,
                                    optimizer_G, optimizer_D, scheduler_G, scheduler_D)
        start_epoch += 1  # 다음 에포크부터 시작

    for epoch in range(start_epoch, EPOCHS+1):
        # tqdm을 이용해 에포크 진행상황 표시
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{EPOCHS}") as pbar:
            for i, batch in pbar:
                real_PDw = batch['PDw'].to(device)
                real_T2w = batch['T2w'].to(device)
                
                current_real_PDw = real_PDw
                current_real_T2w = real_T2w

                # 진짜 및 가짜 레이블 - adjust size for single cross-section
                valid = torch.ones((current_real_PDw.size(0), 1, 16, 16)).to(device)
                fake = torch.zeros((current_real_PDw.size(0), 1, 16, 16)).to(device)

                optimizer_G.zero_grad()

                # GAN loss - using current cross-sections
                fake_PDw = G_T2w2PDw(current_real_T2w)
                loss_GAN_PDw = criterion_GAN(D_PDw(fake_PDw), valid)

                fake_T2w = G_PDw2T2w(current_real_PDw)
                loss_GAN_T2w = criterion_GAN(D_T2w(fake_T2w), valid)

                loss_GAN = (loss_GAN_PDw + loss_GAN_T2w) / 2

                # Cycle consistency loss - using current cross-sections
                recovered_PDw = G_T2w2PDw(fake_T2w)
                loss_cycle_PDw = criterion_cycle(recovered_PDw, current_real_PDw)

                recovered_T2w = G_PDw2T2w(fake_PDw)
                loss_cycle_T2w = criterion_cycle(recovered_T2w, current_real_T2w)

                loss_cycle = (loss_cycle_PDw + loss_cycle_T2w) / 2

                # Total loss
                loss_G = loss_GAN + LAMBDA_CYCLE * loss_cycle
                # loss_G = loss_GAN + LAMBDA_CYCLE * loss_cycle + LAMBDA_IDENTITY * loss_id

                loss_G.backward()
                optimizer_G.step()

                # Discriminator 학습 - using current cross-sections
                optimizer_D.zero_grad()

                loss_real_PDw = criterion_GAN(D_PDw(current_real_PDw), valid)
                loss_fake_PDw = criterion_GAN(D_PDw(fake_PDw.detach()), fake)
                loss_D_PDw = (loss_real_PDw + loss_fake_PDw) / 2
                loss_D_PDw /= 2
                loss_D_PDw.backward()

                loss_real_T2w = criterion_GAN(D_T2w(current_real_T2w), valid)
                loss_fake_T2w = criterion_GAN(D_T2w(fake_T2w.detach()), fake)
                loss_D_T2w = (loss_real_T2w + loss_fake_T2w) / 2
                loss_D_T2w /= 2
                loss_D_T2w.backward()

                optimizer_D.step()

                if i % 100 == 0:
                    pbar.set_postfix(D_loss=(loss_D_PDw + loss_D_T2w).item(), G_loss=loss_G.item())

        scheduler_G.step()
        scheduler_D.step()

        # 체크포인트 저장
        save_checkpoint(epoch, G_PDw2T2w, G_T2w2PDw, D_PDw, D_T2w, optimizer_G, optimizer_D, scheduler_G, scheduler_D)


if __name__ == "__main__":
    print("This is cyclegan.py, but it's not meant to be run directly.")