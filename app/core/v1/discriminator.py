import torch.nn as nn
from torch.nn.utils import spectral_norm
from app.core.self_attention import SelfAttention


class DiscBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DiscBottleneckBlock, self).__init__()
        mid_channels = out_channels // 4

        # Bottleneck 구조 정의 
        self.bottleneck = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            spectral_norm(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            spectral_norm(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=False)),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)),
            )
    
    def forward(self, x):
        x = self.bottleneck(x) + self.shortcut(x)
        return nn.functional.leaky_relu(x, 0.2, inplace=True)


class Discriminator(nn.Module):
    """
    GAN 아키텍쳐에서 판별자(Discriminator)는 진짜 이미지와 생성자(Generator)가 만들어낸 가짜 이미지를 구별하는 역할을 수행.
    일반적인 CNN을 가진 GAN의 판별자는 수용 영역(Receptive Field)의 한계 때문에 국소적인 질감에 집착하지만,
    Self-Attention Layer를 가진 SAGAN의 판별자는 전체적인 구도와 논리적 일관성을 통해 판별한다.

    판별자에 적용된 Spectral Normalization(SN)은 판별자의 기울기(Gradient)가 폭주하는 것을 제한하고,
    립시츠 연속성(Lipschitz Continuity)을 유지하여 안정적인 학습 경로를 제공한다.

    또한, SAGAN은 힌지 손실(Hinge Loss)를 사용하는데, 판별자는 진짜 이미지는 1보다 크게, 가짜 이미지는 -1보다 작게 예측하려고 노력한다.
    이 과정에서 판별자는 단순히 분류하는 것을 넘어, 진짜와 가짜 이미지 사이의 여유 공간(Margin)을 최대화하는 역할을 수행한다.
    """

    def __init__(self, d_conv_dim: int = 64):
        super(Discriminator, self).__init__()

        # (3, 128, 128) -> (64, 64, 64)
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=d_conv_dim, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # (64, 64, 64) -> (128, 32, 32)
        self.layer2 = nn.Sequential(
            DiscBottleneckBlock(in_channels=d_conv_dim, out_channels=d_conv_dim * 2, stride=2)
        )

        # (128, 32, 32) -> (256, 16, 16)
        self.layer3 = nn.Sequential(
            DiscBottleneckBlock(in_channels=d_conv_dim * 2, out_channels=d_conv_dim * 4, stride=2),
            SelfAttention(in_channels=d_conv_dim * 4)
        )

        # (256, 16, 16) -> (512, 8, 8)
        self.layer4 = nn.Sequential(
            DiscBottleneckBlock(in_channels=d_conv_dim * 4, out_channels=d_conv_dim * 8, stride=2)
        )

        # (512, 8, 8) -> (1024, 4, 4)
        self.layer5 = nn.Sequential(
            DiscBottleneckBlock(in_channels=d_conv_dim * 8, out_channels=d_conv_dim * 16, stride=2)
        )

        # Hinge Loss를 사용할 때 선형 출력을 위해 마지막 판정(최종) 레이어에는 SN 적용 X
        # (1024, 4, 4) -> (1, 1, 1) 확률값
        self.final = nn.Sequential(
            nn.Flatten(),    # (Batch, 1024 * 4 * 4)

            spectral_norm(nn.Linear(in_features=d_conv_dim * 16 * 4 * 4, out_features=512, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(in_features=512, out_features=1, bias=False)
        )

    def forward(self, x):
        """
        입력받은 이미지(real or fake)를 연산하여 점수를 산출 

        inputs:
            x: (batch_size, 3, 256, 256) 이미지 데이터

        returns:
            out: (batch_size, 1) 진짜/가짜 판정 점수 
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final(x)

        # (batch_size, 1, 1, 1) -> (batch_size, 1)로 Flatten
        return x.view(x.size(0), -1)