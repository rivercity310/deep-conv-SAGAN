


class SAGANTrainer:
    def __init__(self, generator, discriminator, dataloader, config):
        self.g = generator.to(config.device)
        self.d = discriminator.to(config.device)
        self.dataloader = dataloader 
        self.config = config