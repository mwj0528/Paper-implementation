import torch
import torchvision.transforms as transforms

from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import h5py
from model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_generators_for_inference():
    G_PDw2T2w = Generator().to(device)
    G_T2w2PDw = Generator().to(device)

    # Load the checkpoint file
    checkpoint = torch.load('/mnt/MW/checkpoints/CycleGAN_h5/checkpoint_latest.pth')

    # Access the generator state dictionaries from the checkpoint
    G_PDw2T2w.load_state_dict(checkpoint['G_PDw2T2w_state_dict'])
    G_T2w2PDw.load_state_dict(checkpoint['G_T2w2PDw_state_dict'])

    G_PDw2T2w.eval()
    G_T2w2PDw.eval()

    return G_PDw2T2w, G_T2w2PDw

# 정규화 해제 함수
def denormalize(tensor):
    tensor = (tensor + 1) / 2
    return tensor

import numpy as np

def tensor_to_arr(tensor):
    # 텐서의 차원을 확인한 후 적절한 전처리
    array = tensor.squeeze().cpu().numpy()

    # 만약 3차원 배열이라면 (C, H, W) 형태에서 (H, W, C) 형태로 변환
    if array.ndim == 3:
        return array.transpose(1, 2, 0).astype(np.float32)
    
    # 2차원 배열인 경우 그대로 반환
    return array.astype(np.float32)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < np.finfo(np.float32).eps:
        return float('inf')
    max_pixel = 1.0
    
    return 20 * np.log10(max_pixel/ np.sqrt(mse))

def calculate_ssim(img1, img2):
    min_dim = min(img1.shape[:2])
    win_size = min(11, min_dim // 2)
    win_size = win_size if win_size % 2 == 1 else win_size - 1
    
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        # data_range를 1.0으로 고정 (이미지가 0-1 범위이므로)
        ssim_value = ssim(img1, img2, win_size=win_size, 
                         data_range=1.0, channel_axis=2)
    else:
        ssim_value = ssim(img1, img2, win_size=win_size, 
                         data_range=1.0)
    
    return ssim_value

def prepare_images_for_metrics(real_pil, generated_array):
    """
    Prepare images for metric calculation by ensuring both are in [0,1] range
    """
    # PIL 이미지를 numpy array로 변환하고 0-1 범위로 정규화
    real_array = (np.array(real_pil)/ 255.0).astype(np.float32) 
    
    # 생성된 이미지가 0-255 범위라면 정규화
    if generated_array.max() > 1.0:
        generated_array = (generated_array / 255.0).astype(np.float32) 
        
    return real_array, generated_array

# 모델 불러오기
G_PDw2T2w, G_T2w2PDw = load_generators_for_inference()

# 디렉토리 설정
pdw_dir = "/mnt/MW/data/IXI datasets/h5/test/PDw"
t2w_dir = "/mnt/MW/data/IXI datasets/h5/test/T2w"
output_dir = "/mnt/MW/CycleGAN_h5/test_results"
os.makedirs(output_dir, exist_ok=True)

# PDw 이미지 목록 가져오기 (T2w 폴더와 동일한 파일이 있는 경우만 선택)
pdw_files = sorted(os.listdir(pdw_dir))
t2w_files = sorted(os.listdir(t2w_dir))
common_files = list(set(pdw_files) & set(t2w_files))  # 같은 이름을 가진 파일만 선택

T2w_PSNR_lst = []
T2w_SSIM_lst = []

# 모든 이미지 쌍에 대해 변환 수행
for filename in common_files:
    pdw_path = os.path.join(pdw_dir, filename)
    t2w_path = os.path.join(t2w_dir, filename)

    with h5py.File(pdw_path, 'r') as pdw_data, h5py.File(t2w_path, 'r') as t2w_data:
        sample_image = pdw_data['array'][:]
        gt_image = t2w_data['array'][:]
        
        
        sample_image = ((sample_image - sample_image.min()) / (sample_image.max() - sample_image.min()) * 255).astype(np.uint8)
        gt_image = ((gt_image - gt_image.min()) / (gt_image.max() - gt_image.min()) * 255).astype(np.uint8)
        
        sample_image = Image.fromarray(sample_image.squeeze(), mode='L')
        gt_image = Image.fromarray(gt_image.squeeze(), mode='L')
        
        sample_image_tensor = transform(sample_image).unsqueeze(0).to(device)


    # 모델을 이용한 변환 수행
    with torch.no_grad():
        fake_T2w = G_PDw2T2w(sample_image_tensor)
        recovered_PDw = G_T2w2PDw(fake_T2w)

    # 정규화 해제 [-1,1] → [0,1]
    sample_image_tensor = denormalize(sample_image_tensor)
    fake_T2w = denormalize(fake_T2w)
    recovered_PDw = denormalize(recovered_PDw)
    
    # 텐서를 array로 변환
    sample_pil = tensor_to_arr(sample_image_tensor)
    fake_T2w_pil = tensor_to_arr(fake_T2w)
    recovered_PDw_pil = tensor_to_arr(recovered_PDw)

    # 2x2 시각화 후 저장
    output_path = os.path.join(output_dir, f"PDw_to_T2w_{filename}.png")
    plt.figure(figsize=(8, 8))
    plt.suptitle(f'PDw -> T2w ({filename})', fontweight='bold')

    plt.subplot(2,2,1)
    plt.imshow(sample_pil, cmap= 'gray')
    plt.title("Real PDw")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(gt_image, cmap= 'gray')
    plt.title("Real T2w")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(recovered_PDw_pil, cmap= 'gray')
    plt.title("Recovered PDw")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(fake_T2w_pil, cmap= 'gray')
    plt.title("Fake T2w")
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    real_image, generated_image = prepare_images_for_metrics(gt_image, fake_T2w_pil)

    psnr_value = calculate_psnr(real_image, generated_image)
    ssim_value = calculate_ssim(real_image, generated_image)

    T2w_PSNR_lst.append(psnr_value)
    T2w_SSIM_lst.append(ssim_value)

print(f'PDw -> T2w 평균 PSNR: {np.mean(T2w_PSNR_lst):.2f}, 평균 SSIM: {np.mean(T2w_SSIM_lst):.4f}')


# PDw 이미지 목록 가져오기 (T2w 폴더와 동일한 파일이 있는 경우만 선택)
PDw_PSNR_lst = []
PDw_SSIM_lst = []

# 모든 이미지 쌍에 대해 변환 수행
for filename in common_files:
    pdw_path = os.path.join(pdw_dir, filename)
    t2w_path = os.path.join(t2w_dir, filename)

    with h5py.File(pdw_path, 'r') as pdw_data, h5py.File(t2w_path, 'r') as t2w_data:
        sample_image = t2w_data['array'][:]
        gt_image = pdw_data['array'][:]
        
        sample_image = ((sample_image - sample_image.min()) / (sample_image.max() - sample_image.min()) * 255).astype(np.uint8)
        gt_image = ((gt_image - gt_image.min()) / (gt_image.max() - gt_image.min()) * 255).astype(np.uint8)

        sample_image = Image.fromarray(sample_image.squeeze(), mode='L')
        gt_image = Image.fromarray(gt_image.squeeze(), mode='L')
        
        sample_image_tensor = transform(sample_image).unsqueeze(0).to(device)
    
    # 모델을 이용한 변환 수행
    with torch.no_grad():
        fake_PDw = G_T2w2PDw(sample_image_tensor)
        recovered_T2w = G_PDw2T2w(fake_PDw)

    # 정규화 해제 [-1,1] → [0,1]
    sample_image_tensor = denormalize(sample_image_tensor)
    fake_PDw = denormalize(fake_PDw)
    recovered_T2w = denormalize(recovered_T2w)

    # 텐서를 PIL 이미지로 변환
    sample_pil = tensor_to_arr(sample_image_tensor)
    fake_PDw_pil = tensor_to_arr(fake_PDw)
    recovered_T2w_pil = tensor_to_arr(recovered_T2w)

    # 2x2 시각화 후 저장
    output_path = os.path.join(output_dir, f"T2w_to_PD2w_{filename}.png")
    plt.figure(figsize=(8, 8))
    plt.suptitle(f'T2w -> PDw ({filename})', fontweight='bold')

    plt.subplot(2,2,1)
    plt.imshow(sample_pil, cmap = 'gray')
    plt.title("Real T2w")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(gt_image, cmap = 'gray')
    plt.title("Real PDw")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(recovered_T2w_pil, cmap = 'gray')
    plt.title("Recovered T2w")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(fake_PDw_pil, cmap = 'gray')
    plt.title("Fake PDw")
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    real_image, generated_image = prepare_images_for_metrics(gt_image, fake_PDw_pil)

    psnr_value = calculate_psnr(real_image, generated_image)
    ssim_value = calculate_ssim(real_image, generated_image)

    PDw_PSNR_lst.append(psnr_value)
    PDw_SSIM_lst.append(ssim_value)

print(f'T2w -> PDw 평균 PSNR: {np.mean(PDw_PSNR_lst):.2f}, 평균 SSIM: {np.mean(PDw_SSIM_lst):.4f}')
