import torch.nn as nn
from torch.nn.utils import spectral_norm
from app.core.self_attention import SelfAttention


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

        self.model = nn.Sequential(
            # 입력: (3, 128, 128)
            # 출력: (32, 64, 64)
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=d_conv_dim // 2, kernel_size=4, stride=2,
                                    padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 입력: (32, 64, 64)
            # 출력: (64, 32, 32)
            spectral_norm(nn.Conv2d(in_channels=d_conv_dim // 2, out_channels=d_conv_dim, kernel_size=4, stride=2,
                                    padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 32x32 지점에서 Self-Attention 적용 
            SelfAttention(in_channels=d_conv_dim, allow_sdpa=True),

            # 입력: (64, 32, 32)
            # 출력: (128, 16, 16)
            spectral_norm(nn.Conv2d(in_channels=d_conv_dim, out_channels=d_conv_dim * 2, kernel_size=4, stride=2,
                                    padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 16x16 지점에서 Self-Attention 적용 
            SelfAttention(in_channels=d_conv_dim * 2, allow_sdpa=True),

            # 입력: (128, 16, 16)
            # 출력: (256, 8, 8)
            spectral_norm(nn.Conv2d(in_channels=d_conv_dim * 2, out_channels=d_conv_dim * 4, kernel_size=4, stride=2,
                                    padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 입력: (256, 8, 8)
            # 출력: (512, 4, 4)
            spectral_norm(nn.Conv2d(in_channels=d_conv_dim * 4, out_channels=d_conv_dim * 8, kernel_size=4, stride=2, 
                                    padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 최종 출력은 DCGAN 혹은 SAGAN에서 정석적으로 사용하는 완전 전치 컨볼루션 방식 사용 
            # 4 x 4 크기를 1 x 1로 완전히 수렴시키는 커널 사용 
            # -> 이미지 전체 영역을 아우르는 하나의 커널이 마지막 판정 -> 전역적인 정보를 하나로 응축 
            nn.Conv2d(in_channels=d_conv_dim * 8, out_channels=1, kernel_size=4, stride=1, 
                      padding=0, bias=False)

            # Hinge Loss를 사용할 때 선형 출력을 위해 마지막 판정(최종) 레이어에는 SN 적용 X
        )

    def forward(self, x):
        """
        입력받은 이미지(real or fake)를 연산하여 점수를 산출 

        inputs:
            x: (batch_size, 3, 256, 256) 이미지 데이터

        returns:
            out: (batch_size, 1) 진짜/가짜 판정 점수 
        """
        out = self.model(x)

        # (batch_size, 1, 1, 1) -> (batch_size, 1)로 Flatten
        return out.view(out.size(0), -1)