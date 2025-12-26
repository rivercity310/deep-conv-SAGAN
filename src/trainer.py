import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image


class SAGANTrainer:
    """
    [1. TTUR(Two-Timescale Update Rule)]
    SAGAN 논문에서는 판별자가 너무 느리게 학습되는 것을 방지하기 위해 생성자보다 판별자의 학습률(Learning Rate)을 더 높게 설정.
    ex) g_lr = 0.0001, d_lr = 0.0004

    [2. detach()]
    판별자의 가짜 이미지에 대한 loss를 계산할 때, detach를 적용한 이유는 판별자 학습시 생성자의 가중치까지 미분값이 흐르지 않게 하기 위함.
    """

    def __init__(self, generator, discriminator, dataloader, config):
        self.g = generator.to(config.device)
        self.d = discriminator.to(config.device)
        self.dataloader = dataloader 
        self.config = config

        # TTUR(Two-Timescale Update Rule) 적용 
        self.g_opt = optim.Adam(params=self.g.parameters(), lr=config.g_lr, betas=(0, 0.9))
        self.d_opt = optim.Adam(params=self.d.parameters(), lr=config.d_lr, betas=(0, 0.9))

        self.latent_dim = config.latent_dim
        self.device = config.device

    def train(self, epochs: int):
        for epoch in range(epochs):
            for i, (real_imgs, _) in enumerate(self.dataloader):
                real_imgs = real_imgs.to(self.device)
                b_size = real_imgs.size(0)

                # ============ 판별자 학습 =============
                self.d_opt.zero_grad()

                # 진짜 이미지 손실 (Hinge Loss)
                d_out_real = self.d(real_imgs)
                d_loss_real = nn.ReLU()(1.0 - d_out_real).mean()

                # 가짜 이미지 손실 
                z = torch.randn(b_size, self.latent_dim).to(self.device)
                fake_imgs = self.g(z)
                d_out_fake = self.d(fake_imgs.detach())
                d_loss_fake = nn.ReLU()(1.0 + d_out_fake).mean()

                d_loss = d_loss_real + d_loss_fake 
                d_loss.backward()
                self.d_opt.step()

                # ============ 생성자 학습 ==============
                self.g_opt.zero_grad()

                # 가짜 이미지를 판별자가 진짜로 믿게 만들기 
                g_out_fake = self.d(fake_imgs)
                g_loss = -g_out_fake.mean()

                g_loss.backward()
                self.g_opt.step()

                if i % self.config.log_step == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            # 매 에포크마다 샘플 저장 
            self.save_samples(epoch)

    def save_samples(self, epoch):
        with torch.no_grad():
            z = torch.randn(32, self.latent_dim).to(self.device)
            samples = self.g(z)

            # tanh [-1, 1] -> [0, 1] 복원 
            samples = (samples + 1) / 2
            
            if not os.path.exists('samples'): 
                os.makedirs('samples')
            
            save_image(samples, f"samples/epoch_{epoch}.png", nrow=4)