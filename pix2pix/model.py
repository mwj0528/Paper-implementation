
import os
import time
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(Dataset):
    def __init__(self, image_dir, direction):
        self.image_dir = image_dir
        self.direction = direction
        self.a_path = os.path.join(self.image_dir, "a/")  # 건물 사진 폴더
        self.b_path = os.path.join(self.image_dir, "b/")  # Segmentation 마스크 폴더

        # 이미지 파일 목록 로드 (JPG, PNG 파일만)
        self.image_filenames = [x for x in os.listdir(self.a_path) if x.endswith(('.jpg', '.png'))]

        # 이미지 변환 (256x256 크기 조정 및 정규화)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # [-1, 1]로 정규화
        ])
        self.len = len(self.image_filenames)

    def __getitem__(self, index):
        # 건물 사진 로드
        a_file_path = os.path.join(self.a_path, self.image_filenames[index])
        a = Image.open(a_file_path).convert('RGB')

        # Segmentation mask 로드 (파일 존재 여부 확인)
        b_file_path = os.path.join(self.b_path, self.image_filenames[index])
        if not os.path.exists(b_file_path):
            raise FileNotFoundError(f"Segmentation mask not found: {b_file_path}")
        b = Image.open(b_file_path).convert('RGB')

        # 이미지 변환 적용
        a = self.transform(a)
        b = self.transform(b)

        if self.direction == "a2b":  # 건물 → Segmentation
            return a, b
        else:  # Segmentation → 건물
            return b, a

    def __len__(self):
        return self.len

train_dataset = Dataset("/mnt/MW/data/facades/train/", "b2a")

train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=4, shuffle=True) # Shuffle

def conv(input, output, kernel_size = 4, stride=2, padding=1, batch_norm=True, activation='relu'):
    layers = []

    conv_layer = nn.Conv2d(input, output, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(output))

    # Batch Normalization
    if batch_norm:
        layers.append(nn.BatchNorm2d(output))

    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass

    return nn.Sequential(*layers)

def deconv(input, output, kernel_size=4, stride=2, padding=1, batch_norm=True, activation='relu'):

    layers = []

    deconv_layer = nn.ConvTranspose2d(input, output, kernel_size, stride, padding, bias=False)
    layers.append(deconv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(output))
        
    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == 'none':
        pass

    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = conv(input = 3, output = 64, batch_norm=False, activation='lrelu')
        self.conv2 = conv(input = 64, output = 128, batch_norm=True, activation='lrelu')
        self.conv3 = conv(input = 128, output = 256, batch_norm=True, activation='lrelu')
        self.conv4 = conv(input = 256, output = 512, batch_norm=True, activation='lrelu')
        self.conv5 = conv(input = 512, output = 512, batch_norm=True, activation='lrelu')
        self.conv6 = conv(input = 512, output = 512, batch_norm=True, activation='lrelu')
        self.conv7 = conv(input = 512, output = 512, batch_norm=True, activation='lrelu')
        self.conv8 = conv(input = 512, output = 512, batch_norm=True, activation='relu')

        self.deconv1 = deconv(input = 512, output = 512, batch_norm=True, activation='none')
        self.deconv2 = deconv(input = 1024, output = 512, batch_norm=True, activation='none')
        self.deconv3 = deconv(input = 1024, output = 512, batch_norm=True, activation='none')
        self.deconv4 = deconv(input = 1024, output = 512, batch_norm=True, activation='relu')
        self.deconv5 = deconv(input = 1024, output = 256, batch_norm=True, activation='relu')
        self.deconv6 = deconv(input = 512, output = 128, batch_norm=True, activation='relu')
        self.deconv7 = deconv(input = 256, output = 64, batch_norm=True, activation='relu')
        self.deconv8 = deconv(input = 128, output = 3, batch_norm=False, activation='tanh') # without colorization: output = 2

    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)

        d1 = self.deconv1(e8)
        d1 = F.dropout(d1, 0.5, training=True)
        d1 = F.relu(d1)
        d2 = self.deconv2(torch.cat([d1, e7], 1))
        d2 = F.dropout(d2, 0.5, training=True)
        d2 = F.relu(d2)
        d3 = self.deconv3(torch.cat([d2, e6], 1))
        d3 = F.dropout(d3, 0.5, training=True)
        d3 = F.relu(d3)
        d4 = self.deconv4(torch.cat([d3, e5], 1))
        d5 = self.deconv5(torch.cat([d4, e4], 1))
        d6 = self.deconv6(torch.cat([d5, e3], 1))
        d7 = self.deconv7(torch.cat([d6, e2], 1))
        output = self.deconv8(torch.cat([d7, e1], 1))

        return output


import functools

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator with 70x70 output"""

    def __init__(self, input_nc= 6, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Construct a PatchGAN discriminator.
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last convolution layer
            n_layers (int)  -- the number of convolution layers in the discriminator
            norm_layer      -- normalization layer (default: nn.BatchNorm2d)
        """
        super(Discriminator, self).__init__()
        
        # Check if InstanceNorm is used
        if type(norm_layer) == functools.partial:  
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Kernel size and padding for convolution layers
        kernel_size = 4
        padding = 1

        # Initial layer
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding), 
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Increase the number of filters in each layer
        nf_prev = ndf
        for _ in range(1, n_layers):  # Add intermediate layers
            nf = min(nf_prev * 2, 8 * ndf)  # Gradually increase filters
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(nf),  # Apply normalization
                nn.LeakyReLU(0.2, inplace=True)
            ]
            nf_prev = nf

        # Final layer (output of 70x70 PatchGAN)
        layers += [
            nn.Conv2d(nf_prev, nf_prev * 2, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(nf_prev * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf_prev * 2, 1, kernel_size=kernel_size, stride=1, padding=padding)  # Output 1 channel for PatchGAN
        ]

        # Combine layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        """Forward pass through the discriminator"""
        return self.model(input)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

G = Generator().to(device).apply(weights_init_normal)
D = Discriminator().to(device).apply(weights_init_normal)

criterionL1 = nn.L1Loss().to(device)
criterionBCE = nn.BCEWithLogitsLoss().to(device)

g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

def save_checkpoint(epoch, G, D, g_optimizer, d_optimizer):
    """모델 체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }
    save_path = os.path.join('/mnt/MW/checkpoints/pix2pix/', f'checkpoint_latest.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: checkpoint_latest.pth")

def load_checkpoint(checkpoint_path, G, D, g_optimizer, d_optimizer):
    """모델 체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path)

    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch


# 학습을 위한 설정
num_epochs = 100
lambda_pixel = 100  # 픽셀 손실 가중치
patch = (1, 70, 70)  # Discriminator의 출력 크기 설정


# 학습 시간 기록
start_time = time.time()

# 손실 값 기록용 딕셔너리
loss_hist = {'G': [], 'D': []}

def train():
    start_epoch = 0
    if os.path.exists('/mnt/MW/checkpoints/pix2pix/checkpoint_latest.pth'):
        start_epoch = load_checkpoint('/mnt/MW/checkpoints/pix2pix/checkpoint_latest.pth', G, D, g_optimizer, d_optimizer)
        start_epoch += 1  # 다음 에포크부터 시작

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss_G = 0
        epoch_loss_D = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        
        for i, (real_a, real_b) in progress_bar:
            ba_si = real_a.size(0)

            real_a = real_a.to(device)
            real_b = real_b.to(device)

            # 생성된 가짜 이미지
            fake_b = G(real_a)

            # Discriminator 진짜/가짜 판별
            in_dis_real = torch.cat((real_b, real_a), 1)
            out_dis_real = D(in_dis_real)
            real_loss = criterionBCE(out_dis_real, torch.ones_like(out_dis_real))

            in_dis_fake = torch.cat((fake_b.detach(), real_a), 1)
            out_dis_fake = D(in_dis_fake)
            fake_loss = criterionBCE(out_dis_fake, torch.zeros_like(out_dis_fake))

            # Generator 손실 계산
            g_loss = criterionBCE(D(torch.cat((fake_b, real_a), 1)), torch.ones_like(out_dis_real)) + lambda_pixel * criterionL1(fake_b, real_b)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Discriminator 손실 계산
            d_loss = (real_loss + fake_loss) / 2
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 손실 기록
            epoch_loss_G += g_loss.item()
            epoch_loss_D += d_loss.item()

            # 진행률 업데이트
            progress_bar.set_postfix(G_Loss=f"{g_loss.item():.6f}", D_Loss=f"{d_loss.item():.6f}")

        avg_loss_G = epoch_loss_G / len(train_loader)
        avg_loss_D = epoch_loss_D / len(train_loader)
        epoch_time = (time.time() - epoch_start_time) / 60
        print(f"\n[Epoch {epoch}/{num_epochs}] G_Loss: {avg_loss_G:.6f}, D_Loss: {avg_loss_D:.6f}, Time: {epoch_time:.2f} min")

        save_checkpoint(epoch, G, D, g_optimizer, d_optimizer)