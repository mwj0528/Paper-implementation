import torch
import torchvision
from torch.utils.data import DataLoader
from model import Generator, Discriminator, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

generator = Generator()
discriminator = Discriminator()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

checkpoint_path = '/mnt/MW/checkpoints/pix2pix/checkpoint_latest.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['G_state_dict'])
discriminator.load_state_dict(checkpoint['D_state_dict'])
optimizer_g.load_state_dict(checkpoint['g_optimizer_state_dict'])
optimizer_d.load_state_dict(checkpoint['d_optimizer_state_dict'])
print(f"Checkpoint loaded from {checkpoint_path}")

transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # [-1, 1]로 정규화
        ])

# 모델을 평가 모드로 설정
generator.eval()
discriminator.eval()

# 테스트 데이터셋 로드 (여기서는 가상의 CustomDataset 사용)
test_dataset = Dataset("/mnt/MW/data/facades/test/", "b2a")  # 실제 경로와 변환 적용
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def denormalize(output):
    output = (output + 1) / 2
    return output
    
# 테스트 수행
with torch.no_grad():  # 기울기 계산 비활성화
    total_loss = 0
    for i, data in enumerate(test_loader):
        if i > 5:
            break
        
        real_images, _ = data  # 실제 이미지 데이터

        real_images = real_images.to(device)  # GPU로 데이터 이동

        # Generator로 이미지 생성
        fake_images = generator(real_images)

        real_output = denormalize(real_images.squeeze().cpu()).permute(1, 2, 0)
        fake_output = denormalize(fake_images.squeeze().cpu()).permute(1, 2, 0)

        # 시각화 후 저장
        output_path = os.path.join("/mnt/MW/pix2pix/results", f"result_{i}.png")
        plt.figure(figsize=(8, 8))

        plt.subplot(2,1,1)
        plt.imshow(real_output)
        plt.title("input image")
        plt.axis('off')

        plt.subplot(2,1,2)
        plt.imshow(fake_output)
        plt.title("translated image")
        plt.axis('off')

        plt.savefig(output_path, bbox_inches='tight')
        plt.close()