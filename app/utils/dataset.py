import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def rgb_loader(path):
    """
    이미지를 읽어와서 무조건 3채널 RGB로 변환 
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_dataloader(root_dir, batch_size, image_size):
    """
    DataLoader를 생성하여 반환하는 헬퍼 함수 

    [1. Normalize]
    Generator의 마지막 출력에 하이퍼볼릭 탄젠트 활성화 함수를 사용했기 때문에 출력값이 [-1, 1] 범위.
    따라서 실제 이미지도 이 범위로 맞추어야 판별자가 진짜와 가짜를 동일 기준에서 비교할 수 있음.
    ex) (원본값 - 0.5) / 0.5 => 이 연산을 통해 0은 -1이 되고, 1은 1이 됨 

    [2. RandomHorizontalFlip]
    사진을 좌우로 반전시켜 모델이 새로운 사진으로 인식하게 하여 학습 데이터가 2배로 늘어나는 효과(Data Augmentation) 기대.
    """
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(size=image_size),
        transforms.RandomHorizontalFlip(p=0.5),        # 데이터 증강(좌우 반전)
        transforms.ToTensor(),                         # [0, 1] 범위로 변환 
        transforms.Normalize((0.5, 0.5, 0.5),          # [0, 1] -> [-1, 1] 변환 
                                (0.5, 0.5, 0.5))
    ])

    root_dir = os.path.abspath(os.path.join(os.getcwd(), root_dir))
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,              # 매 에폭마다 순서 섞기 
        pin_memory=True,           # GPU 전송 속도 향상,
        drop_last=True
    )