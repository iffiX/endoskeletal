import numpy as np
import torch as t
import torch.nn as nn
import lightning.pytorch as pl
from tqdm import tqdm
from .encoder import RobotConvEncoder
from .decoder import RobotConvDecoder
from synthetic.sample import StarRobot


class StarVAE(pl.LightningModule):
    def __init__(
        self,
        e_dim: int,
        min_num_nodes: int,
        max_num_nodes: int,
        grid_size: int,
        vrn_dim: int,
        vrn_depth: int,
        conv_layers: int,
        beta: float = 0.1,
        lr: float = 3e-5,
        step_size: int = 4,
        gamma: float = 0.91,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        self.encoder = RobotConvEncoder(
            f_dim=max_num_nodes + 2,
            e_dim=e_dim,
            grid_size=grid_size,
            vrn_dim=vrn_dim,
            vrn_depth=vrn_depth,
            n_conv_encoder_layers=conv_layers,
        )
        self.decoder = RobotConvDecoder(
            f_dim=max_num_nodes + 2,
            e_dim=e_dim,
            grid_size=grid_size,
            vrn_dim=vrn_dim,
            vrn_depth=vrn_depth,
            n_conv_t_decoder_layers=conv_layers,
        )

    def get_samples(self, n, seed=None):
        samples = StarRobot(
            self.hparams.min_num_nodes,
            self.hparams.max_num_nodes,
            n,
            self.device,
            seed,
            self.hparams.grid_size,
        ).get()
        return samples

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        x_ = self.decoder(z)
        return x_

    def rsample(self, mu, logvar):
        z_dist = t.distributions.Normal(loc=mu, scale=t.exp(logvar * 0.5))
        return z_dist.rsample()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.rsample(mu, logvar)
        x_ = self.decode(z)
        return x_

    def training_step(self, batch, batch_idx):
        x = self.get_samples(len(batch))
        mu, logvar = self.encode(x)
        z = self.rsample(mu, logvar)
        x_ = self.decode(z)
        kld = -0.5 * t.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_loss = kld.mean()
        recon_loss = self.decoder_loss(x, x_)
        loss = recon_loss + self.beta * kld_loss
        self.log(
            "reconstruction_loss",
            recon_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "kld_loss",
            kld_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_samples(len(batch))
        mu, logvar = self.encode(x)
        z = self.rsample(mu, logvar)
        x_ = self.decode(z)
        kld = -0.5 * t.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_loss = kld.mean()
        recon_loss = self.decoder_loss(x, x_)
        self.log(
            "val_reconstruction_loss",
            recon_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_kld_loss",
            kld_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        metrics = {"val_loss": recon_loss}
        self.log_dict(metrics, sync_dist=True)
        return metrics

    def on_train_epoch_end(self):
        self.beta = min(self.beta * 1.12, 1)

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = t.optim.AdamW(params, lr=self.hparams.lr)
        scheduler = t.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def decoder_loss(self, x, x_):
        labels = t.argmax(x, dim=1)
        ce_loss = nn.CrossEntropyLoss()(x_, labels)
        cew_loss = nn.CrossEntropyLoss(weight=self.get_class_weight(x))(x_, labels)
        return ce_loss + cew_loss

    def generate_by_latent(self, latent):
        self.decoder.eval()
        with t.no_grad():
            if type(latent) is not t.Tensor:
                latent = t.tensor(np.array(latent), dtype=t.float32).to(self.device)
            new_samples = self.decode(latent)
        return new_samples

    def generate(self, batch_size, seed=None, mean=None, var=None):
        if seed is None:
            latent = t.randn([batch_size, self.hparams.e_dim], device=self.device)
        else:
            generator = t.Generator(self.device).manual_seed(seed)
            latent = t.randn(
                [batch_size, self.hparams.e_dim],
                generator=generator,
                device=self.device,
            )
        if mean is not None and var is not None:
            latent = latent * var + mean
        return self.generate_by_latent(latent)

    @staticmethod
    def get_class_weight(target_segmentation):
        seg_num = t.sum(target_segmentation, dim=(0, 2, 3, 4))
        weight = t.where(seg_num > 0, 1 / seg_num, 0)
        return weight / weight.sum()

    def generate_mean_var(self, batch_size):
        t.use_deterministic_algorithms(True)
        latents = []
        with t.no_grad():
            for _ in tqdm(range(batch_size), total=batch_size):
                Xs = self.get_samples(1)
                mu, logvar = self.encode(Xs)
                latents.append(self.rsample(mu, logvar))
            latent = t.cat(latents, dim=0)
        vae_mean = t.mean(latent, dim=0).detach().cpu()
        vae_var = t.var(latent, dim=0).detach().cpu()
        t.use_deterministic_algorithms(False)
        return vae_mean, vae_var
