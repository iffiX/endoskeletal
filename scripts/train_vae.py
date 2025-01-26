import torch as t
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from model.vae.star_vae import StarVAE
from synthetic.dataset import PlaceholderDataset


robot_dataset = PlaceholderDataset(10000000)

train_size = int(0.9 * len(robot_dataset))
valid_size = len(robot_dataset) - train_size
train_dataset, valid_dataset = random_split(robot_dataset, [train_size, valid_size])
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    num_workers=0,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    num_workers=0,
    shuffle=False,
)

vae = StarVAE(
    e_dim=512,
    min_num_nodes=3,
    max_num_nodes=8,
    grid_size=64,
    vrn_dim=32,
    vrn_depth=5,
    conv_layers=3,
)
vae.hparams.lr = 3e-5
vae.beta = 1
checkpoint_callback = ModelCheckpoint(
    dirpath="../data/ckpt/vae/",
    filename="v2-{epoch:02d}-{val_loss:.3f}",
    save_top_k=10,
    verbose=True,
    monitor="val_loss",
    mode="min",
    every_n_epochs=1,
)
logger = TensorBoardLogger(save_dir="logger")
trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    limit_train_batches=4096,
    limit_val_batches=512,
    strategy="ddp",
    devices=[4, 5, 6, 7],
    check_val_every_n_epoch=1,
    max_epochs=-1,
)
t.set_float32_matmul_precision("high")
trainer.fit(vae, train_loader, valid_loader)
