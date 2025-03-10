import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os
import glob
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = '/mnt/MW/data/edges2shoes/'
output_dir = '/mnt/MW/MUNIT/test_results'
n_samples = 200
n_styles = 8

# 모델 정의 (학습 스크립트와 동일)
class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
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

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
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

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ='relu'):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
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
    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
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
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2, style_dim=8):
        super(Encoder, self).__init__()
        self.content_encoder = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.style_encoder = StyleEncoder(in_channels, dim, n_downsample, style_dim)
    
    def forward(self, x):
        content = self.content_encoder(x)
        style = self.style_encoder(x)
        return content, style

class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_residual=3, n_upsample=2, style_dim=8):
        super(Decoder, self).__init__()
        layers = []
        dim = dim * 2 ** n_upsample
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")]
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, content_code, style_code):
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img

# 테스트용 데이터셋
class TestDataset(Dataset):
    def __init__(self, root, transforms_=None, domain="A"):
        self.transform = transforms_
        self.domain = domain  # 'A' 또는 'B' 도메인 선택
        
        # 테스트 폴더에서 파일 로드
        self.files = sorted(glob.glob(os.path.join(root, "val") + "/*.*"))

    def __getitem__(self, index):
        # 이미지 열기
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        
        # 도메인에 따라 이미지 자르기
        if self.domain == "A":
            img_domain = img.crop((0, 0, w / 2, h))  # 왼쪽 절반 (A 도메인)
        else:  # domain == "B"
            img_domain = img.crop((w / 2, 0, w, h))  # 오른쪽 절반 (B 도메인)

        # 변환 적용 (무작위 좌우 반전은 테스트에서 제외)
        img_domain = self.transform(img_domain)

        return img_domain

    def __len__(self):
        return len(self.files)
def denormalize(tensor):
    return (tensor + 1) / 2

def load_generators_for_inference():
    Enc1 = Encoder().to(device)
    Enc2 = Encoder().to(device)
    Dec1 = Decoder().to(device)
    Dec2 = Decoder().to(device)
    
    checkpoint = torch.load('/mnt/MW/checkpoints/MUNIT/checkpoint_latest.pth', map_location=device)
    
    # 엄격하지 않은 로드로 테스트
    Enc1.load_state_dict(checkpoint['Enc1_state_dict'], strict=False)
    Enc2.load_state_dict(checkpoint['Enc2_state_dict'], strict=False)
    Dec1.load_state_dict(checkpoint['Dec1_state_dict'], strict=False)
    Dec2.load_state_dict(checkpoint['Dec2_state_dict'], strict=False)

    Enc1.eval()
    Enc2.eval()
    Dec1.eval()
    Dec2.eval()

    return Enc1, Enc2, Dec1, Dec2

# 모델 초기화
Enc1, Enc2, Dec1, Dec2 = load_generators_for_inference()

# 테스트용 변환
transforms_ = [
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

# 테스트 데이터 로드
test_data_A = TestDataset(data_root, transforms_ = transforms.Compose(transforms_), domain='A')
test_data_B = TestDataset(data_root, transforms_ = transforms.Compose(transforms_), domain='B')
# 테스트 수행
n_samples = min(n_samples, len(test_data_A))

with torch.no_grad():
    for i in range(n_samples):
        img_A = test_data_A[i].unsqueeze(0).to(device)
        c_A, _ = Enc1(img_A)
        results = [img_A]
        for j in range(n_styles):
            s_B = torch.randn(1, 8, 1, 1).to(device)
            fake_B = Dec2(c_A, s_B)
            results.append(fake_B)
        result = torch.cat(results, 3)
        save_image(denormalize(result), 
                   f"{output_dir}/A2B_sample_{i}.png", 
                   nrow=1, padding=2, normalize=False)
    
    for i in range(n_samples):
        img_B = test_data_B[i].unsqueeze(0).to(device)
        c_B, _ = Enc2(img_B)
        results = [img_B]
        for j in range(1):
            s_A = torch.randn(1, 8, 1, 1).to(device)
            fake_A = Dec1(c_B, s_A)
            results.append(fake_A)
        result = torch.cat(results, 3)
        save_image(denormalize(result), 
                   f"{output_dir}/B2A_sample_{i}.png", 
                   nrow=1, padding=2, normalize=False)

print(f"테스트 완료. 결과는 {output_dir} 디렉토리에 저장되었습니다.")