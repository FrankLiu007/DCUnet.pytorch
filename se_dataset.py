from tqdm import tqdm
import torch
import sac


class RfDataset(torch.utils.data.Dataset):

    def read_data(self, file_list):
        print("reading all data to memory.....")
        data = {"z_cmp": [], "rf": []}
        sampling_rate = self.pars["sampling_rate"]
        for z_cmp, rf in file_list:
            head_z, z = sac.read_sac(z_cmp)
            head_rf, rf = sac.read_sac(z_cmp)
            if round(1 / head_z['delta']) != sampling_rate or round(1 / head_rf['delta']) != sampling_rate:
                print("Sampling rates of Z component or receiver function are not the same with pars...")
                print("sampling rate in parameters:", sampling_rate, "sampling rate of z:", round(1 / head_z['delta']),
                      "sampling rate of rf:", round(1 / head_rf['delta']))
                print("at present, the resampling method was not implemented yet")
                continue
            cutted_z = self.cut_data(head_z, z)
            cutted_rf = self.cut_data(head_rf, rf)
            data["z_cmp"].append(cutted_z)
            data["rf"].append(cutted_rf)
        return data

    def cut_data(self, head, data):
        b = self.pars["begin_time"]
        e = self.pars["stop_time"]
        b_sample = round((b - head["b"]) / head["delta"]) + 1
        e_sample = round((e - head["b"]) / head["delta"]) + 1
        return data[b_sample:e_sample]

    def __init__(self, args, file_list):
        self.pars = args
        self.file_list=file_list
        self.dataset = self.read_data(file_list)
        self.length = len(self.dataset["rf"])

    def __getitem__(self, idx):
        z_cmp = torch.from_numpy(self.dataset['z_cmp'][idx]).type(torch.FloatTensor)
        rf = torch.from_numpy(self.dataset['rf'][idx]).type(torch.FloatTensor)

        return z_cmp, rf

    def __len__(self):
        return self.length
