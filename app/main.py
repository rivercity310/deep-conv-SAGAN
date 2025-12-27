import yaml
import torch.nn as nn
from pathlib import Path
from app.utils.dataset import get_dataloader
from app.utils.trainer import SAGANTrainer
from app.core.discriminator import Discriminator
from app.core.generator import Generator


# 설정값 로드 
config_path = Path.cwd() / "configs" / "config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 상수 
PATH = config["path"]
DATA_DIR = PATH["data_dir"]

MODEL = config["model"]
LATENT_DIM = MODEL["latent_dim"]
IMAGE_SIZE = MODEL["image_size"]
G_CONV_DIM = MODEL["g_conv_dim"]
D_CONV_DIM = MODEL["d_conv_dim"]

TRAIN = config["train"]
EPOCHS = TRAIN["epochs"]
BATCH_SIZE = TRAIN["batch_size"]

print(config)

# 데이터 로더 준비 
dataloader = get_dataloader(
    root_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)


# 가중치 초기화 함수 
def weights_init(m):
    """
    훈련을 시작하는 출발점을 정하기 위해 모델의 초기 가중치 세팅.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # SN이 적용된 경우 weight_orig를 초기화해야 함
        if hasattr(m, 'weight_orig'):
            nn.init.orthogonal_(m.weight_orig.data, 1.0)
        else:
            nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)

# 모델 
generator = Generator(latent_dim=LATENT_DIM, g_conv_dim=G_CONV_DIM)
generator.apply(weights_init)
discriminator = Discriminator(d_conv_dim=D_CONV_DIM)
discriminator.apply(weights_init)

# 트레이너 실행 
trainer = SAGANTrainer(
    generator=generator,
    discriminator=discriminator,
    dataloader=dataloader,
    config=config
)

trainer.train(epochs=EPOCHS)