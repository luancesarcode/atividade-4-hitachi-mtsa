import torch
from mtsa.utils.utils import Wav2Array, Array2Mfcc
from torch.utils.data import Dataset


class FeatureCNNDataset(Dataset):
    def __init__(self, raw_audios, sampling_rate=16000, is_score=False, device="cuda:0"):
        super().__init__()

        self.raw_audios = raw_audios
        self.sampling_rate = sampling_rate
        self.is_score = is_score
        self.device = device
        
        self.wav2array = Wav2Array(sampling_rate=self.sampling_rate)
        self.array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)


    def __len__(self):
        return len(self.raw_audios)

    def __getitem__(self, idx):
        file_path = self.raw_audios[idx]
        x_array = self.wav2array.transform([file_path])
        x_array_tensor = torch.from_numpy(x_array).squeeze(0).float().to(self.device)


        if self.is_score:
            return x_array_tensor, x_array_tensor
        
        with torch.no_grad():
            target_mfcc = self.array2mfcc.transform(x_array).squeeze(0)
            target_mfcc = torch.from_numpy(target_mfcc).to(self.device)

            
        return x_array_tensor, target_mfcc