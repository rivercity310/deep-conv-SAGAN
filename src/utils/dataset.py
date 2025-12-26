import os 
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SAGANDataset(Dataset):
    """
    [1. Normalize]
    Generator의 마지막 출력에 하이퍼볼릭 탄젠트 활성화 함수를 사용했기 때문에 출력값이 [-1, 1] 범위.
    따라서 실제 이미지도 이 범위로 맞추어야 판별자가 진짜와 가짜를 동일 기준에서 비교할 수 있음.
    ex) (원본값 - 0.5) / 0.5 => 이 연산을 통해 0은 -1이 되고, 1은 1이 됨 

    [2. RandomHorizontalFlip]
    사진을 좌우로 반전시켜 모델이 새로운 사진으로 인식하게 하여 학습 데이터가 2배로 늘어나는 효과(Data Augmentation) 기대.
    """

    def __init__(self, root_dir, image_size=256):
        """
        root_dir: 이미지가 저장된 폴더 경로 
        image_size: 모델이 요구하는 해상도 (256x256)
        """
        self.root_dir = root_dir 
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        
        # 이미지 전처리 파이프라인 
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),        # 데이터 증강(좌우 반전)
            transforms.ToTensor(),                         # [0, 1] 범위로 변환 
            transforms.Normalize((0.5, 0.5, 0.5),          # [0, 1] -> [-1, 1] 변환 
                                 (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        # GAN 학습시 레이블은 보통 필요 없으므로 0 고정 반환 
        return image, 0     
    

def get_dataloader(root_dir, batch_size, image_size=256, num_workers=4):
    """
    DataLoader를 생성하여 반환하는 헬퍼 함수 
    """
    dataset = SAGANDataset(root_dir, image_size=image_size)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,             # 매 에폭마다 순서 섞기 
        num_workers=num_workers,  # 병렬 데이터 로딩
        pin_memory=True           # GPU 전송 속도 향상 
    )