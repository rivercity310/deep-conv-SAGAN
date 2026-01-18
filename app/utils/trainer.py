import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
from app.core.v1.generator import Generator
from app.core.v1.discriminator import Discriminator


class SAGANTrainer:
    """
    [1. TTUR(Two-Timescale Update Rule)]
    SAGAN 논문에서는 판별자가 너무 느리게 학습되는 것을 방지하기 위해 생성자보다 판별자의 학습률(Learning Rate)을 더 높게 설정.
    ex) g_lr = 0.0001, d_lr = 0.0004

    [2. detach()]
    판별자의 가짜 이미지에 대한 loss를 계산할 때, detach를 적용한 이유는 판별자 학습시 생성자의 가중치까지 미분값이 흐르지 않게 하기 위함.
    """

    def __init__(self, generator: Generator, discriminator: Discriminator, dataloader, config):
        # 상수 
        self.history = {"d_loss": [], "g_loss": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        TRAIN = config["train"]
        self.sample_step = TRAIN["sample_step"]
        self.checkpoint_step = TRAIN["checkpoint_step"]

        MODEL = config["model"]
        self.latent_dim = MODEL["latent_dim"]

        PATH = config["path"]
        self.sample_dir = PATH["sample_dir"]
        self.checkpoint_dir = PATH["checkpoint_dir"]

        self.g = generator.to(self.device)
        self.d = discriminator.to(self.device)
        self.dataloader = dataloader 

        # TTUR(Two-Timescale Update Rule) 적용 
        betas = (TRAIN["beta1"], TRAIN["beta2"])
        self.g_opt = optim.Adam(params=self.g.parameters(), lr=TRAIN["lr_g"], betas=betas)
        self.d_opt = optim.Adam(params=self.d.parameters(), lr=TRAIN["lr_d"], betas=betas)

        self.fixed_noise = torch.randn(32, self.latent_dim).to(self.device)

    def train(self, epochs: int):
        sample_dir = os.path.abspath(os.path.join(os.getcwd(), self.sample_dir))

        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        for epoch in range(epochs):
            d_running_loss = 0.0
            g_running_loss = 0.0
            progress_bar = tqdm(enumerate(self.dataloader),
                                total=len(self.dataloader),
                                desc=f"Epoch [{epoch + 1} / {epochs}]")

            for i, (real_imgs, _) in progress_bar:
                real_imgs = real_imgs.to(self.device)
                b_size = real_imgs.size(0)

                # ============ 판별자 학습 =============
                self.d_opt.zero_grad()

                # 진짜 이미지 손실 (Hinge Loss)
                # - 판별자에 Label Smoothing 적용 (무조건 100% 가짜라고 확신하지 못하게 함)
                # - 훈련과정에서 D_LOSS가 0.0에 수렴하여 판별자가 완벽하게 구분한다면 생성자 입장에서는 기울기(Gradient)를 전혀 받지 못함.
                # - 따라서 아무리 학습을 해도 실력이 늘지 않는 기울기 소실(Gradient Vanishing) 상태에 빠짐.
                d_out_real = self.d(real_imgs)
                d_loss_real = nn.ReLU()(1.0 - d_out_real).mean()

                # 가짜 이미지 손실 
                z = torch.randn(b_size, self.latent_dim).to(self.device)
                fake_imgs = self.g(z)
                d_out_fake = self.d(fake_imgs.detach())
                d_loss_fake = nn.ReLU()(1.0 + d_out_fake).mean()

                d_loss = (d_loss_real + d_loss_fake) / 2

                d_loss.backward()
                self.d_opt.step()

                d_running_loss += d_loss.item()

                # ============ 생성자 학습 ==============
                # D_LOSS = 0 문제(판별자 승리 문제) 해결을 위해 n번씩 생성자 훈련  
                self.g_opt.zero_grad()

                # 가짜 이미지를 판별자가 진짜로 믿게 만들기 
                z = torch.randn(b_size, self.latent_dim).to(self.device)
                fake_imgs_new = self.g(z)

                g_out_fake = self.d(fake_imgs_new)
                g_loss = -g_out_fake.mean()

                g_loss.backward()
                self.g_opt.step()

                g_running_loss += g_loss.item()

                # tqdm 진행바 오른쪽에 실시간 Loss 값 표시 
                progress_bar.set_postfix({
                    "D_LOSS": f"{d_loss.item():.4f}",
                    "G_LOSS": f"{g_loss.item():.4f}"
                })
            
            # 고정 노이즈 이미지 생성 
            fixed_noise_path = os.path.join(sample_dir, f"fixed_noise_{epoch + 1}.png")
            random_noise_path = os.path.join(sample_dir, f"random_noise_{epoch + 1}.png")

            self.g.eval()
            with torch.no_grad():
                # 고정 노이즈 이미지
                fake_img = self.g(self.fixed_noise).detach().cpu()
                save_image(fake_img, fixed_noise_path, normalize=True, value_range=(-1, 1))

                # 랜덤 노이즈 이미지 
                z = torch.randn(32, self.latent_dim).to(self.device)
                fake_img_random = self.g(z).detach().cpu()
                save_image(fake_img_random, random_noise_path, normalize=True, value_range=(-1, 1))

            self.g.train()

            if (epoch + 1) % self.checkpoint_step == 0:
                self.save_checkpoint(epoch)

            # 매 애폭 종료 후 Loss 기록 및 그래프 업데이트 
            avg_d = d_running_loss / len(self.dataloader)
            avg_g = g_running_loss / len(self.dataloader)
            self.history["d_loss"].append(avg_d)
            self.history["g_loss"].append(avg_g)

            self.save_loss_plot(epoch + 1)

    def save_loss_plot(self, epoch):
        """학습 진행 상황을 그래프로 저장"""
        plt.figure(figsize=(10, 5))
        plt.title(f"SAGAN Training Loss (Epoch {epoch})")

        epochs_range = range(1, len(self.history["g_loss"]) + 1)
        # Generator Loss
        plt.plot(epochs_range, self.history["g_loss"], label="Generator Loss", color='tab:red', linewidth=2)
        # Discriminator Loss
        plt.plot(epochs_range, self.history["d_loss"], label="Discriminator Loss", color='tab:blue', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        
        # y축 로그 스케일 (G_LOSS가 너무 높을 경우를 대비 - 선택 사항)
        # 만약 G_LOSS가 너무 커서 D_LOSS가 일직선으로 보인다면 아래 주석을 해제하세요.
        # plt.yscale('log') 

        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # x축 단위를 정수로 표시 (에폭이 적을 때 유용)
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plot_path = os.path.join(self.sample_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()

    def load_checkpoint(self, path):
        """저장된 체크포인트로부터 학습 재개"""
        print(f"🔄 Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.g.load_state_dict(checkpoint['g_state_dict'])
        self.d.load_state_dict(checkpoint['d_state_dict'])
        self.g_opt.load_state_dict(checkpoint['g_opt_state_dict'])
        self.d_opt.load_state_dict(checkpoint['d_opt_state_dict'])
        
        return checkpoint['epoch']
        
    def save_checkpoint(self, epoch):
        """
        모델 및 옵티마이저 상태 저장 
        """
        checkpoint_dir = os.path.abspath(os.path.join(os.getcwd(), self.checkpoint_dir))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 저장할 상태 딕셔너리 구성 
        state = {
            "epoch": epoch,
            "g_state_dict": self.g.state_dict(),
            "d_state_dict": self.d.state_dict(),
            "g_opt_state_dict": self.g_opt.state_dict(),
            "d_opt_state_dict": self.d_opt.state_dict(),
            "gamma_g": self.get_gamma_values(self.g),
            "gamma_d": self.get_gamma_values(self.d)
        }

        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_sa32_{epoch + 1}.pth")
        torch.save(state, path)
        print(f"Epoch {epoch + 1} - Checkpoint Saved: {path}")

    def get_gamma_values(self, model):
        gammas = []
        for name, param in model.named_parameters():
            if "gamma" in name:
                gammas.append(param.item())

        return gammas