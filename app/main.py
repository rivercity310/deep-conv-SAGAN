import yaml
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

# 모델 
generator = Generator(latent_dim=LATENT_DIM, g_conv_dim=G_CONV_DIM)
discriminator = Discriminator(d_conv_dim=D_CONV_DIM)

# 트레이너 실행 
trainer = SAGANTrainer(
    generator=generator,
    discriminator=discriminator,
    dataloader=dataloader,
    config=config
)

trainer.train(epochs=EPOCHS)