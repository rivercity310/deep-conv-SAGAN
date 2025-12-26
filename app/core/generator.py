import torch.nn as nn 
from torch.nn.utils import spectral_norm
from app.core.self_attention import SelfAttention


class Generator(nn.Module):
    """
    GAN 아키텍쳐에서 Generator(생성자)는 판별자(Discriminator)의 피드백을 통해 가중치 업데이트를 수행.

    ConvTranspose(전치 행렬곱) 연산:
        작은 해상도의 feature_map을 큰 해상도의 이미지로 확장할 때 사용하는 연산으로 UpSampling에 해당.
        일반적인 Convolution 연산이 여러 픽셀을 모아 하나의 값을 만든다면, ConvTranspose 연산은 하나의 픽셀값을
        커널(Kernel) 모양대로 넓게 펼쳐 뿌리는 작업.

        Convolution 연산은 입력 x에 행렬 C를 곱해 출력 y를 만드는 과정으로, 이때 행렬 C는 정보를 압축하는 행렬.
        ex) y = Cx

        ConvTranspose는 C를 Transpose한 C^T를 곱하여 차원이 거꾸로 커지는 효과를 구현.
        ex) y' = C^Tx'

        [1] 연산과정 
            1. 입력 픽셀 선택 
            2. 그 값(스칼라)에 커널(가중치)을 곱함
            3. 커널 크기만큼 feature map의 해당 위치에 그 값을 복사(뿌리기)
            4. 옆 픽셀로 이동해 반복하고, 겹치는 부분은 더함
        
        [2] 전치 행렬곱에서의 stride 적용
            Convolution 연산에서 stride는 n만큼 건너뛰며 정보를 압축하는 것이지만, ConvTranspose에서 stride는 입력 픽셀 사이에
            0을 삽입하여 공간을 강제로 늘리는 것을 의미
            ex) 2x2 입력 데이터 사이에 0을 넣어 3x3 혹은 그 이상으로 부풀림 

        [3] 전치 행렬곱에서의 padding 적용 
            Convolution 연산에서 패딩은 가장자리에 0을 추가하는 것으로, 출력을 키우는 역할 수행.
            ConvTranspose에서 패딩은 가장자리를 잘라내는(Crop) 역할 수행.
            ex) padding=1인 경우 계산된 결과물의 테두리를 1칸씩 버림 

        ConvTranspose는 stride로 인해 부풀려진 입력 데이터(x) 위로 커널이 지나가며 Convolution 연산 수행.
        이 과정을 통해 0으로 채워졌던 빈칸들이 의미 있는 특징값들로 채워지게 됨.            

    Spectral Normalization 적용:
        [1] Exploding Gradients 억제
            판별자(Discriminator)가 매우 강력해서 모든 가짜를 다 잡아낸다면 생성자에게 전달되는 기울기(Gradient) 값이 너무 커져서 모델이 망가짐.
            SN은 각 레이어의 가중치 행렬이 가질 수 있는 최대 변화율(Spectral Norm)을 1로 제한하여 학습이 안정적으로 진행될 수 있도록 함.

        [2] Self Attention Layer 호환성 
            만약 Generator의 특정 가중치가 너무 큰 경우 특정 픽셀 관계에만 과도하게 attention 하게 되어서 이미지가 깨짐.

        [3] Mode Collapse 방지
            생성자가 특정 스타일만 계속 그리는 현상(모드 붕괴)은 생성자가 판별자를 속이기 위해 특정 가중치를 비정상적으로 키울 때 자주 발생함.
            SN을 적용하면 생성자가 가중치를 무한정 키울 수 없응므로, 더 넓은 범위의 상상력(Latent Space)를 탐색.
            즉, 다양성 확보 측면에서 유리함.
    """

    def __init__(self, latent_dim: int, g_conv_dim: int = 64):
        """
        latent_dim: 잠재공간의 차원 수, 즉 n개의 추상적인 지표 (압축된 정보)
        초기 latent_dim 잠재공간이 ConvTranspose2d를 거치며 점점 커짐
        """
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # 입력: (latent_dim, 1, 1)
            # 출력: (512, 4, 4)
            spectral_norm(nn.ConvTranspose2d(in_channels=latent_dim, out_channels=g_conv_dim * 8, kernel_size=4,
                                             stride=1, padding=0, bias=False)),
            nn.ReLU(inplace=True),

            # 입력: (512, 4, 4)
            # 출력: (256, 8, 8)
            spectral_norm(nn.ConvTranspose2d(in_channels=g_conv_dim * 8, out_channels=g_conv_dim * 4, kernel_size=4, stride=2,
                                             padding=1, bias=False)),
            nn.ReLU(inplace=True),

            # 입력: (256, 8, 8)
            # 출력: (128, 16, 16)
            spectral_norm(nn.ConvTranspose2d(in_channels=g_conv_dim * 4, out_channels=g_conv_dim * 2, kernel_size=4, stride=2,
                                             padding=1, bias=False)),
            nn.ReLU(inplace=True),

            # 16x16 해상도 지점에서 Self-Attention 적용 
            SelfAttention(in_channels=g_conv_dim * 2),

            # 입력: (128, 16, 16)
            # 출력: (64, 32, 32)
            spectral_norm(nn.ConvTranspose2d(in_channels=g_conv_dim * 2, out_channels=g_conv_dim, kernel_size=4, stride=2,
                                             padding=1, bias=False)),
            nn.ReLU(inplace=True),

            # 32x32 해상도 지점에서 Self-Attention 적용 
            SelfAttention(in_channels=g_conv_dim),

            # 입력: (64, 32, 32)
            # 출력: (32, 64, 64)
            spectral_norm(nn.ConvTranspose2d(in_channels=g_conv_dim, out_channels=g_conv_dim // 2, kernel_size=4, stride=2,
                                             padding=1, bias=False)),
            nn.ReLU(inplace=True),

            # 입력: (32, 64, 64)
            # 출력: (16, 128, 128)
            spectral_norm(nn.ConvTranspose2d(in_channels=g_conv_dim // 2, out_channels=g_conv_dim // 4, kernel_size=4, stride=2,
                                             padding=1, bias=False)),
            nn.ReLU(inplace=True),

            # 입력: (16, 128, 128)
            # 출력: (3, 256, 256)
            spectral_norm(nn.ConvTranspose2d(in_channels=g_conv_dim // 4, out_channels=3, kernel_size=4, stride=2,
                                             padding=1, bias=False)),
            nn.Tanh()
        )

    def forward(self, z):
        """
        inputs:
            z: (batch, latent_dim) -> (batch, latent_dim, 1, 1)
        """
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)
