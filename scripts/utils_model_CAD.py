import os
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim

import json
from datetime import datetime

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################################################
########################################################
def get_anomaly_score(recon_batch,data_batch):
    # recon_batch = recon_.cpu().detach().numpy()
    # data_batch = data_.cpu().detach().numpy()
    data_batch = data_batch.to(device)
    recon_batch = recon_batch.to(device)
    # print(data_batch.shape, recon_batch.shape)
    abs_diff = torch.abs(recon_batch - data_batch)
    mean_diff = abs_diff.mean(dim=1)
    max_score = mean_diff.max(dim=-1).values.max(dim=-1).values
    return max_score

############################################################################
# Reparameterization trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
############################################################################
def get_reconstructed(Enc,Dec,data_,device='cuda'):
    data_ = Variable(data_).to(device)
    mu, logvar = Enc(data_)
    z = reparameterize(mu, logvar)
    recon_ = Dec(z)
    del data_, mu,logvar,z
    return recon_#.to(device)

############################################################################
def model_override(model_path, suffix):
    model_path_ = os.path.join(model_path, f"model_{suffix}.pt")
    new_model = os.path.join(model_path, f"model_{suffix}_old.pt")
    if os.path.exists(model_path_):
        os.rename(model_path_, new_model)
        print(f'file renamed from  ')
        print(model_path_,'->', new_model)
    else:
        print(f'Path not exist ', model_path_)


########################################################################################################################################################

# Encoder
class Encoder(nn.Module):
    def __init__(self, z_size=64):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512 * 8 * 8, z_size)
        self.fc_logvar = nn.Linear(512 * 8 * 8, z_size)

    def forward(self, x):
        h = self.conv_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, z_size=64):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_size, 512 * 8 * 8)
        self.deconv_layers = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),      # 64x64 -> 128x128
            nn.Tanh()  # Output scaled between -1 and 1
        )

    def forward(self, z):
        h = self.fc(z)
        h = self.deconv_layers(h)
        return h

# Define the model for blur/fake detection
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

########################################################################################################################################################
# def save_model(Enc, Dec, D, optEncDec, optD, paths, loss_history, suffix, verbose = False):
#     # last_saved_epoch = epoch  # Track the epoch at which the model was saved
#     model_path = os.path.join(paths.path_models, f"model_{suffix}.pt")
#     if verbose:
#         print(f'SAVING MODEL AT -> {model_path}')
#     torch.save({
#         'encoder_state_dict':       Enc.state_dict(),
#         'decoder_state_dict':       Dec.state_dict(),
#         'discriminator_state_dict': D.state_dict(),
#         'optimizer_enc_state_dict': optEncDec.state_dict(),
#         'optimizer_dec_state_dict': optD.state_dict(),
#         'loss_history':            pd.DataFrame(loss_history)
#         }, model_path)
    
#     # print(f"Model saved at epoch {epoch} to {model_path}")
#     # return last_saved_epoch

# def load_model(Enc, Dec, D, optEncDec, optD, paths, suffix,device='cuda', verbose = False):
#     model_path = os.path.join(paths.path_models, f"model_{suffix}.pt")
#     if verbose:
#         print(f'TRYING MODEL FROM -> {model_path}')
#     if not os.path.exists(model_path): # model does not exists
#         if verbose:
#             print(f' path {model_path} --> {os.path.exists(model_path)} ')
#         return []
    
#     checkpoint = torch.load(model_path,
#                              map_location=device,
#                              weights_only=False        # allow non-weight objects (e.g., pandas DF)
#                              )
    
#     Enc.load_state_dict      (checkpoint['encoder_state_dict'])
#     Dec.load_state_dict      (checkpoint['decoder_state_dict'])
#     D.load_state_dict(checkpoint['discriminator_state_dict'])
#     optEncDec.load_state_dict(checkpoint['optimizer_enc_state_dict'])
#     optD.load_state_dict(checkpoint['optimizer_dec_state_dict'])
#     loss_history    = checkpoint['loss_history'].to_dict('records')
#     if verbose:
#         print(f"Model loaded at epochs: {len(loss_history)} ({model_path})")

#     # print(f"Model loaded from epoch {last_saved_epoch}")
#     return loss_history

def safe_transform_to_string(transform_obj):
    """Convert augmentation/transform pipeline to readable string."""
    try:
        return str(transform_obj)
    except Exception:
        return "Unavailable"

def save_model(
    Enc,
    Dec,
    D,
    optEncDec,
    optD,
    loss_history,
    path_models,
    suffix,
    epoch,
    train_loader=None,
    val_loader=None,
    params=None,
    augmentation=None,
    dataset_name=None,
    train_dir=None,
    notes=None,
    verbose=False
):
    os.makedirs(path_models, exist_ok=True)

    model_path = os.path.join(path_models, f"model_{suffix}.pt")
    config_path = os.path.join(path_models, f"model_{suffix}_config.json")

    # -----------------------------
    # dataset / loader statistics
    # -----------------------------
    n_train_images = None
    n_val_images = None
    batch_size = None

    if train_loader is not None:
        batch_size = getattr(train_loader, "batch_size", None)
        try:
            n_train_images = len(train_loader.dataset)
        except Exception:
            n_train_images = None

    if val_loader is not None:
        try:
            n_val_images = len(val_loader.dataset)
        except Exception:
            n_val_images = None

    # -----------------------------
    # convert loss history safely
    # -----------------------------
    if isinstance(loss_history, pd.DataFrame):
        loss_history_to_save = loss_history
    else:
        loss_history_to_save = pd.DataFrame(loss_history)

    # -----------------------------
    # build metadata/config
    # -----------------------------
    config = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "VAE-GAN",
        "suffix": suffix,
        "epoch": epoch,
        "epochs_trained": len(loss_history_to_save),

        "dataset": {
            "dataset_name": dataset_name,
            "train_dir": train_dir,
            "n_train_images": n_train_images,
            "n_val_images": n_val_images,
        },

        "training": {
            "batch_size": batch_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },

        "augmentation": {
            "pipeline": safe_transform_to_string(augmentation)
        },

        "params": params if params is not None else {},
        "notes": notes,
    }

    # -----------------------------
    # save checkpoint
    # -----------------------------
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": Enc.state_dict(),
        "decoder_state_dict": Dec.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "optimizer_enc_state_dict": optEncDec.state_dict(),
        "optimizer_dec_state_dict": optD.state_dict(),
        "loss_history": loss_history_to_save,
        "config": config,
    }

    torch.save(checkpoint, model_path)

    # save separate JSON config for readability
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    if verbose:
        print(f"Saved model checkpoint to: {model_path}")
        print(f"Saved config JSON to:      {config_path}")
        print(f"Epoch: {epoch}")
        print(f"Train images: {n_train_images}")
        print(f"Val images: {n_val_images}")
        print(f"Epochs trained: {len(loss_history_to_save)}")

#############################################################################################################

def load_model(Enc, Dec, D, optEncDec, optD, path_models, suffix, verbose=False, device='cuda'):
    model_path = os.path.join(path_models, f"model_{suffix}.pt")

    if verbose:
        print(f"TRYING MODEL FROM -> {model_path}")

    if not os.path.exists(model_path):
        if verbose:
            print(f"Path does not exist: {model_path}")
        return [], None

    checkpoint = torch.load(model_path, map_location=device)

    Enc.load_state_dict(checkpoint['encoder_state_dict'])
    Dec.load_state_dict(checkpoint['decoder_state_dict'])
    D.load_state_dict(checkpoint['discriminator_state_dict'])
    optEncDec.load_state_dict(checkpoint['optimizer_enc_state_dict'])
    optD.load_state_dict(checkpoint['optimizer_dec_state_dict'])

    loss_history_raw = checkpoint.get('loss_history', [])
    if hasattr(loss_history_raw, "to_dict"):
        loss_history = loss_history_raw.to_dict('records')
    else:
        loss_history = loss_history_raw

    config = checkpoint.get("config", None)
    last_saved_epoch = checkpoint.get("epoch", len(loss_history))

    if verbose:
        print(f"Loaded model from {model_path}")
        print(f"Model loaded at epoch {last_saved_epoch}")
        # print(f"Model saved at  {config.get('saved_at', {})}")   

        if config is not None:
            print("Loaded config:")

            print(f"  Dataset name: {config.get('dataset', {}).get('dataset_name', None)}")
            print(f"  Train images: {config.get('dataset', {}).get('n_train_images', None)}")
            print(f"  Val images:   {config.get('dataset', {}).get('n_val_images', None)}")
            print(f"  Batch size:   {config.get('training', {}).get('batch_size', None)}")
            print(f"  Augment:      {config.get('augmentation', {}).get('pipeline', None)}")

    return loss_history, config

#############################################################################################################
############################################################################


def get_loss_functions (verbose=True):
    reconstruction_loss_fn = nn.MSELoss()  # L2 reconstruction loss
    adversarial_loss_fn = nn.BCEWithLogitsLoss()  # For GAN
    if verbose:
        print("Loss functions initialized:")
        print("Reconstruction Loss Function: ", reconstruction_loss_fn)
        print("Adversarial Loss Function: ", adversarial_loss_fn)
    return reconstruction_loss_fn, adversarial_loss_fn
############################################################################


def get_optimizers (Enc,Dec,Dis,learning_rate_enc_dec=0.001,learning_rate_dis=0.0001, verbose=True):
    optEncDec = optim.Adam(list(Enc.parameters()) + list(Dec.parameters()), lr=learning_rate_enc_dec)
    optDis = optim.Adam(Dis.parameters(), lr=learning_rate_dis)
    return optEncDec, optDis