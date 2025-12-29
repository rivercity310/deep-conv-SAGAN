import torch
import torch.nn as nn
import torch.nn.functional as F 


class SelfAttention(nn.Module):
    """
    Convolution Layer 통과 후 생성된 feature_map을 입력받아 Self-Attention 작업을 수행하는 레이어.

    배경:
        [1]
        Convolution 연산은 receptive field가 이미지의 local에 해당하기 때문에
        long range dependency를 모델링하기 위해서는 여러 convolution layer를 통과한 후에 처리가 가능함.

        [2] 
        그렇다고 kernel의 크기를 증가시킨다면 receptive field가 커지니 표현용량을 키울 수 있지만
        local convolution 구조를 사용해 얻은 픽셀에 대한 계산 정보가 흐릿해짐. -> 정보 손실 위험 

    효용성:
        [1]
        Self Attention Layer(Module)은 모든 위치에서 feature의 가중치 합으로 위치 반응을 계산하며
        이를 통해 long range dependency 문제를 해결함.

        [2]
        즉, 입력값 일부에 대해 입력값 전체에 대한 관계를 계산함으로써, 이미지의 경우 픽셀 하나와 이미지 전체가
        임베딩된 feature map 사이의 관계를 계산하는 연산을 진행함. 
        => 각각의 값(픽셀)들이 다른 값(픽셀)들과 얼마나 관련되어 있는지 입력값 내의 연관성 계산 

    연산 과정:
        [1]
        직전 convolution layer를 통과한 feature map이 입력으로 들어오면 
        해당 feature map으로 Query, Key, Value 각각을 1x1 convolution으로 계산.

        [2]
        생성된 Query와 Key의 행렬곱 연산 및 Softmax 함수를 통해 attention map 생성 
        계산된 attention map은 query와 key 값들의 연관된 정도에 따라 다른 밝기 수준을 가짐.
        => 관련성이 높으면 밝게, 낮으면 어둡게 표현됨.
        => Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V

        [3]
        생성된 attention map을 Value와 행렬곱 및 1x1 convolution 연산하여 self-attention feature map(o) 계산 
        SAGAN에서는 기존 Attention 연산에서 o를 생성하기 위해 1x1 convolution 연산이 추가됨.

        [4]
        최종 결과는 o와 입력값 x를 더해 출력. 이때 학습 가능한 스칼라 값 0으로 초기화된 gamma를 o에 곱함.
        네트워크가 처음에는 local 주변 신호에 의존하다 점차 학습이 진행되며 non-local 신호에 더 많은 가중치를 부여하는 방법으로 학습.
        즉, 초기에 무작위한 값을 뱉어내는 Self Attention 연산을 출력에 반영하면 학습이 불안정해지므로 점진적으로 적용하기 위함.
        => y_i = gamma * o_i + x_i
    """

    def __init__(self, in_channels: int, k: int = 8, allow_sdpa: bool = False):
        super(SelfAttention, self).__init__()
        self.emb_channels = in_channels // k
        self.allow_sdpa = allow_sdpa
        self.key = nn.Conv2d(in_channels=in_channels, out_channels=self.emb_channels, kernel_size=1)
        self.query = nn.Conv2d(in_channels=in_channels, out_channels=self.emb_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_channels, out_channels=self.emb_channels, kernel_size=1)
        self.self_att = nn.Conv2d(in_channels=self.emb_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.softmax = nn.Softmax(dim=-1)
        self.last_attn_map = None

    def forward(self, x, return_attn=False):
        """
        inputs:
            x: 직전 Convolution Layer를 통과한 feature map (batch_size, channel, width, height)
            return_attn: Attention Map 반환 여부 (시각화를 위해)
        
        returns:
            y: residual connection
        """
        # C는 in_channels, C'는 emb_channels를 나타냄 
        batch_size, C, W, H = x.size()
        N = W * H                                                           # 총 Feature 개수

        if not self.allow_sdpa or return_attn:
            # Flatten 작업: (B, C, W, H) -> (B, C, N)
            q = self.query(x).view(batch_size, -1, N).permute(0, 2, 1)        # Query: (B, N, C')
            k = self.key(x).view(batch_size, -1, N)                           # Key: (B, C', N)
            v = self.value(x).view(batch_size, -1, N)                         # Value: (B, C', N)

            # (B, N, C') * (B, C', N) => (B, N, N): N x N Attention Map 생성 
            s = torch.bmm(q, k) / (self.emb_channels ** 0.5)
            beta = self.softmax(s)
            self.last_attn_map = beta.detach().cpu()

            # Value에 Attention 적용 (V * Attention_Map^T)
            # (B, C', N) * (B, N, N) => (B, C', N)
            v = torch.bmm(v, beta.permute(0, 2, 1))

            # v를 다시 (B, C', W, H) -> (B, C, W, H)로 복원 
            v = v.view(batch_size, -1, W, H)
            o = self.self_att(v)

        else:
            # Q, K, V 생성 및 SDPA 형식으로 변환 (L = N, E = emb_channels)
            # SDPA 기대 형식: (B, H< L< E) -> 여기서 헤드(H)는 1로 설정 
            q = self.query(x).view(batch_size, self.emb_channels, N).permute(0, 2, 1).unsqueeze(1)    
            k = self.key(x).view(batch_size, self.emb_channels, N).permute(0, 2, 1).unsqueeze(1)
            v = self.value(x).view(batch_size, self.emb_channels, N).permute(0, 2, 1).unsqueeze(1)

            # scale 인자를 따로 주지 않으면 기본값인 sqrt(d_k) 적용
            # 이는 값이 너무 커지는 것을 막아 softmax가 한쪽으로 쏠리는 Saturation 현상 완화
            attn_out = F.scaled_dot_product_attention(q, k, v)   # (B, 1, N, C')

            # 다시 (B, C', W, H)로 복원 
            o = attn_out.squeeze(1).permute(0, 2, 1).view(batch_size, self.emb_channels, W, H)
            o = self.self_att(o)

        # 잔차 연결(Residual Connection) 적용 -> 처음부터 어텐션을 강하게 사용하면 학습이 망가짐 
        # 학습 초기(gamma가 0에 가까울 때)는 기존 Convolution 결과(x)만 사용하다가 
        # 점차 모델이 gamma 값을 키워가면서 Attention 값을 반영함 
        y = self.gamma * o + x 
        return y
