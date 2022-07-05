import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm

import utils
from models.unet import Unet
from models.layers.istft import ISTFT
from se_dataset import AudioDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='exp/unet16.json', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
args = parser.parse_args()
torch.cuda.set_device(1)

n_fft, hop_length = 400, 160
window = torch.hann_window(n_fft).cuda()
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()

def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1) # Batch preserving sum for convenience.
    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


def main():

    params = utils.Params(args.model_dir)
    losses=[]
    net = Unet(params.model).cuda()

    train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='val')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
            collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
            collate_fn=test_dataset.collate, shuffle=False, num_workers=4)

    torch.set_printoptions(precision=10, profile="full")

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # Learning rate scheduler
    scheduler = ExponentialLR(optimizer, 0.95)

    for epoch in range(args.num_epochs):
        train_bar = tqdm(train_data_loader)
        for input in train_bar:
            z_cmp, rf, seq_len = map(lambda x: x.cuda(), input)

            mixed = stft(z_cmp).unsqueeze(dim=1)
            real, imag = z_cmp[..., 0], z_cmp[..., 1]
            out_real, out_imag = net(real, imag)
            out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
            rf_predicted = istft(out_real, out_imag, z_cmp.size(1))
            rf_predicted = torch.squeeze(rf_predicted, dim=1)
            for i, l in enumerate(seq_len):
                rf_predicted[i, l:] = 0

            loss = wSDRLoss(z_cmp, rf, rf_predicted)
            losses.append(loss.numpy()[0])
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        scheduler.step()
    torch.save(net.state_dict(), './final.pth.tar')

    print( losses )


if __name__ == '__main__':
    main()
