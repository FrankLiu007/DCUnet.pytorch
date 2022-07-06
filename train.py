import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
import pickle

import utils
from models.unet import Unet
from models.layers.istft import ISTFT
from se_dataset import RfDataset
import datetime


def wSDRLoss(z_cmp, rf, rf_predicted, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1)  # Batch preserving sum for convenience.

    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = z_cmp - rf
    noise_est = z_cmp - rf_predicted

    a = bsum(rf ** 2) / (bsum(rf ** 2) + bsum(noise ** 2) + eps)
    wSDR = a * mSDRLoss(rf, rf_predicted) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


def test(model, device, test_loader, stft, istft):
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for train_data, target in test_loader:
            train_data, target = train_data.to(device), target.to(device)
            data = stft(train_data).unsqueeze(dim=1)
            real, imag = data[..., 0], data[..., 1]
            out_real, out_imag = model(real, imag)
            out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
            output = istft(out_real, out_imag, train_data.size(1))
            output = torch.squeeze(output, dim=1)
            loss = wSDRLoss(train_data, target, output)
            losses.append(loss.item())

    return losses



def train(args, model, device, train_loader, optimizer, epoch, stft, istft):
    model.train()
    losses = []
    for train_data, target in tqdm(train_loader):
        train_data, target = train_data.to(device), target.to(device)

        data = stft(train_data).unsqueeze(dim=1)
        real, imag = data[..., 0], data[..., 1]
        out_real, out_imag = model(real, imag)
        out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
        output = istft(out_real, out_imag, train_data.size(1))
        output = torch.squeeze(output, dim=1)

        loss = wSDRLoss(train_data, target, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return  losses


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Receiver function')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 1.0e-3)')

    parser.add_argument('--device', type=int, default=0,
                        help='using CUDA for training, -1 is cpu, 0 is cuda 0 etc.')

    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--dataset_path', default="dataset.lst", metavar='path',
                        help='path to dataset file lst, containing train and test file list ')

    parser.add_argument('--model_path', default="exp/unet16.json", metavar='path',
                        help='path to dataset file lst, containing train and test file list ')

    parser.add_argument('--sampling_rate', type=int, default=20, metavar='sr',
                        help='sampling rate for all waveforms (default: 20)')
    parser.add_argument('--begin_time', type=float, default=-5, metavar='sr',
                        help='begin time of the window used to cut waveforms (default: -5)')
    parser.add_argument('--stop_time', type=float, default=90, metavar='sr',
                        help='stop time of the window used to cut waveforms (default: -90)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    device = None
    if args.device != -1:
        device = torch.device(args.device)
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        device = torch.device("cpu")

    #### load file list
    f = open(args.dataset_path, 'rb')
    file_list = pickle.load(f)

    test_file_list = []
    for value in file_list["test"].values():
        test_file_list = test_file_list + value
    test_file_list = test_file_list[:100]

    train_file_list = []
    for value in file_list["train"].values():
        train_file_list = train_file_list + value
    train_file_list = train_file_list[:1000]

    train_kwargs["file_list"] = train_file_list
    train_set = RfDataset(args, train_file_list)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=4)

    train_kwargs["file_list"] = test_file_list
    test_set = RfDataset(args, test_file_list)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    params = utils.Params(args.model_path)
    Net = Unet(params.model)
    model = Net.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = ExponentialLR(optimizer, 0.95)

    n_fft, hop_length = 512, 160
    window = torch.hann_window(n_fft).to(device)
    stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
    istft = ISTFT(n_fft, hop_length, window='hanning').to(device)

    train_losses = []
    test_losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch, stft, istft)
        train_losses.append(loss)
        loss = test(model, device, test_loader, stft, istft)
        test_losses.append(loss)

        scheduler.step()
    prefix = str(datetime.datetime.now())

    print("Writing model out....")
    torch.save(model.state_dict(), prefix + ".dcunet.pt")

    print("Writing losses out....")
    f = open(prefix + ".losses", 'wb')
    pickle.dump({"train": train_losses, "test": test_losses}, f)
    f.close()


if __name__ == '__main__':
    main()
