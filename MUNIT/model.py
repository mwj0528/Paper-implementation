import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
import random
import os
import itertools
from tqdm import tqdm
import glob
import numpy as np

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터 설정
BATCH_SIZE = 1
EPOCHS = 200
LR = 0.0001
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_GAN = 1.0
LAMBDA_x = 10.0
LAMBDA_c = 1.0
LAMBDA_s = 1.0
LAMBDA_cyc = 1.0

# Dataset 정의의
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))
                              
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))
        
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')
            
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return {'A': img_A, 'B': img_B}
    
    def __len__(self):
        return len(self.files)
    
transforms_ = [
    transforms.Resize((128,128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataloader = DataLoader(
    ImageDataset('/mnt/MW/data/edges2shoes/', transforms_=transforms_),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8
)
 
# MUNIT 모델 정의

def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)
    
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, features, norm = "in"):
        super(ResidualBlock, self).__init__()
        
        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features)
        )
        
    def forward(self, x):
        return x + self.block(x)

# Encoder  
class ContentEncoder(nn.Module):
    def __init__(self, in_channels = 3, dim = 64, n_residual = 3, n_downsample = 2):
        super(ContentEncoder, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        ]
        
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim*2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim*2),
                nn.ReLU(inplace=True)
            ]
            dim *= 2
            
        for _ in range(n_residual):
            layers += [ResidualBlock(dim)]
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class StyleEncoder(nn.Module):
    def __init__(self, in_channels = 3, dim = 64, n_downsample = 2, style_dim = 8):
        super(StyleEncoder, self).__init__()
        
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.ReLU(inplace=True)]
        
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim*2, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            dim *= 2
            
        for _ in range(n_downsample - 2):
            layers += [nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0)]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, in_channels = 3, dim = 64, n_residual = 3, n_downsample = 2, style_dim = 8):
        super(Encoder, self).__init__()
        
        self.content_encoder = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.style_encoder = StyleEncoder(in_channels, dim, n_downsample, style_dim)
        
    def forward(self, x):
        content = self.content_encoder(x)
        style = self.style_encoder(x)
        
        return content, style

# MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim = 256, n_blk = 3, activ = 'relu'):
        super(MLP, self).__init__()
        
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
            
        layers += [nn.Linear(dim, output_dim)]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# Decoder
class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_residual=3, n_upsample=2, style_dim=8):
        super(Decoder, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*layers)

        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, content_code, style_code):
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img

# Discriminator
class MultiDisciminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDisciminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                )
            )
            
        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)
        
    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
    
def save_checkpoint(epoch, Enc1, Dec1,
                   Enc2, Dec2, D1, D2,
                   optimizer_G, optimizer_D1, optimizer_D2,
                   schedular_G, schedular_D1, schedular_D2):
    """모델 체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'Enc1_state_dict': Enc1.state_dict(),
        'Enc2_state_dict': Enc2.state_dict(),
        'Dec1_state_dict': Dec1.state_dict(),
        'Dec2_state_dict': Dec2.state_dict(),
        'D1_state_dict': D1.state_dict(),
        'D2_state_dict': D2.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D1_state_dict': optimizer_D1.state_dict(),
        'optimizer_D2_state_dict': optimizer_D2.state_dict(),
        'scheduler_G_state_dict': schedular_G.state_dict(),
        'scheduler_D1_state_dict': schedular_D1.state_dict(),
        'scheduler_D2_state_dict': schedular_D2.state_dict()
    }
    save_dir = "/mnt/MW/checkpoints/MUNIT/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'checkpoint_latest.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: checkpoint_latest.pth")

def load_checkpoint(checkpoint_path, Enc1, Enc2, Dec1, Dec2, D1, D2, 
                    optimizer_G, optimizer_D1, optimizer_D2, schedular_G, schedular_D1, schedular_D2):
    """모델 체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path)

    Enc1.load_state_dict(checkpoint['Enc1_state_dict'])
    Enc2.load_state_dict(checkpoint['Enc2_state_dict'])
    Dec1.load_state_dict(checkpoint['Dec1_state_dict'])
    Dec2.load_state_dict(checkpoint['Dec2_state_dict'])
    D1.load_state_dict(checkpoint['D1_state_dict'])
    D2.load_state_dict(checkpoint['D2_state_dict'])
    
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D1.load_state_dict(checkpoint['optimizer_D1_state_dict'])
    optimizer_D2.load_state_dict(checkpoint['optimizer_D2_state_dict'])
    schedular_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
    schedular_D1.load_state_dict(checkpoint['scheduler_D1_state_dict'])
    schedular_D2.load_state_dict(checkpoint['scheduler_D2_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch

Enc1 = Encoder(dim = 64, n_residual = 3, n_downsample = 2, style_dim = 8).to(device)
Enc2 = Encoder(dim = 64, n_residual = 3, n_downsample = 2, style_dim = 8).to(device)
Dec1 = Decoder(dim = 64, n_residual = 3, n_upsample = 2, style_dim = 8).to(device)
Dec2 = Decoder(dim = 64, n_residual = 3, n_upsample = 2, style_dim = 8).to(device)
D1 = MultiDisciminator().to(device)
D2 = MultiDisciminator().to(device)

Enc1.apply(weight_init_normal)
Enc2.apply(weight_init_normal)
Dec1.apply(weight_init_normal)
Dec2.apply(weight_init_normal)
D1.apply(weight_init_normal)
D2.apply(weight_init_normal)

optimizer_G = optim.Adam(itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()), lr=LR, betas=(BETA1, BETA2))
optimizer_D1 = optim.Adam(D1.parameters(), lr=LR, betas=(BETA1, BETA2))
optimizer_D2 = optim.Adam(D2.parameters(), lr=LR, betas=(BETA1, BETA2))

schedular_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(EPOCHS, 0, 100).step)
schedular_D1 = lr_scheduler.LambdaLR(optimizer_D1, lr_lambda=LambdaLR(EPOCHS, 0, 100).step)
schedular_D2 = lr_scheduler.LambdaLR(optimizer_D2, lr_lambda=LambdaLR(EPOCHS, 0, 100).step)

loss_l1 = nn.L1Loss().to(device)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def train():
    valid = 1
    fake = 0
    
    start_epoch = 1
        
    if os.path.exists('/mnt/MW/checkpoints/MUNIT/checkpoint_latest.pth'):
        start_epoch = load_checkpoint('/mnt/MW/checkpoints/MUNIT/checkpoint_latest.pth',
                                        Enc1, Dec1, Enc2, Dec2, D1, D2, 
                                        optimizer_G, optimizer_D1, optimizer_D2, schedular_G, schedular_D1, schedular_D2)
        start_epoch += 1  # 다음 에포크부터 시작
        
    for epoch in range(start_epoch, EPOCHS+1):
        # tqdm을 이용해 에포크 진행상황 표시
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{EPOCHS}") as pbar:
            for i, batch in pbar:
                X1 = Variable(batch["A"].type(Tensor))
                X2 = Variable(batch["B"].type(Tensor))

            # Sampled style codes
                style_1 = Variable(torch.randn(X1.size(0), 8, 1, 1).type(Tensor))
                style_2 = Variable(torch.randn(X1.size(0), 8, 1, 1).type(Tensor))

                # -------------------------------
                #  Train Encoders and Generators
                # -------------------------------

                optimizer_G.zero_grad()

                # Get shared latent representation
                c_code_1, s_code_1 = Enc1(X1)
                c_code_2, s_code_2 = Enc2(X2)

                # Reconstruct images
                X11 = Dec1(c_code_1, s_code_1)
                X22 = Dec2(c_code_2, s_code_2)

                # Translate images
                X21 = Dec1(c_code_2, style_1)
                X12 = Dec2(c_code_1, style_2)

                # Cycle translation
                c_code_21, s_code_21 = Enc1(X21)
                c_code_12, s_code_12 = Enc2(X12)
                X121 = Dec1(c_code_12, s_code_1) if LAMBDA_cyc > 0 else 0
                X212 = Dec2(c_code_21, s_code_2) if LAMBDA_cyc > 0 else 0

                # Losses
                loss_GAN_1 = LAMBDA_GAN * D1.compute_loss(X21, valid)
                loss_GAN_2 = LAMBDA_GAN * D2.compute_loss(X12, valid)
                loss_ID_1 = LAMBDA_x * loss_l1(X11, X1)
                loss_ID_2 = LAMBDA_x * loss_l1(X22, X2)
                loss_s_1 = LAMBDA_s * loss_l1(s_code_21, style_1)
                loss_s_2 = LAMBDA_s * loss_l1(s_code_12, style_2)
                loss_c_1 = LAMBDA_c * loss_l1(c_code_12, c_code_1.detach())
                loss_c_2 = LAMBDA_c * loss_l1(c_code_21, c_code_2.detach())
                loss_cyc_1 = LAMBDA_cyc * loss_l1(X121, X1) if LAMBDA_cyc > 0 else 0
                loss_cyc_2 = LAMBDA_cyc * loss_l1(X212, X2) if LAMBDA_cyc > 0 else 0

                # Total loss
                loss_G = (
                    loss_GAN_1
                    + loss_GAN_2
                    + loss_ID_1
                    + loss_ID_2
                    + loss_s_1
                    + loss_s_2
                    + loss_c_1
                    + loss_c_2
                    
                    + loss_cyc_1
                    + loss_cyc_2
                )

                loss_G.backward()
                optimizer_G.step()

                # -----------------------
                #  Train Discriminator 1
                # -----------------------

                optimizer_D1.zero_grad()

                loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)

                loss_D1.backward()
                optimizer_D1.step()

                # -----------------------
                #  Train Discriminator 2
                # -----------------------

                optimizer_D2.zero_grad()

                loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)

                loss_D2.backward()
                optimizer_D2.step()

                if i % 100 == 0:
                    pbar.set_postfix(D_loss=(loss_D1 + loss_D2).item(), G_loss=loss_G.item())
        # Update learning rates
        schedular_G.step()
        schedular_D1.step()
        schedular_D2.step()

        save_checkpoint(epoch, Enc1, Enc2, Dec1, Dec2, D1, D2, optimizer_G, optimizer_D1, optimizer_D2, schedular_G, schedular_D1, schedular_D2)
        
        
if __name__ == "__main__":
    print("This is MUNIT training script.")