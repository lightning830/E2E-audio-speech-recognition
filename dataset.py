import os
import glob
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import sys
from scipy.io import wavfile
import librosa
from torch.utils.data import Dataset
from specaugment import freq_mask, time_mask


class MyDataset(Dataset):
    def __init__(self, datasets, datadir, charToIx, audioParams, train=True):
        super(MyDataset, self).__init__()

        # dataset path list
        self.datalist = []

        for dataset in datasets:
            with open(str(datadir) + "/" + str(dataset) + ".txt", "r") as f:
                lines = f.readlines()

            # pretrain
            if 'pre' in dataset:
                for line in lines:
                    self.datalist.append(str(datadir) + "/pretrain/" + line.strip())
            # main, val, test
            else:
                for line in lines:
                    self.datalist.append(str(datadir) + "/main/" + line.strip())

        self.charToIx = charToIx
        self.dataset = dataset
        self.audioParams = audioParams
        self.train = train
        return

    def __getitem__(self, index):
        # audioFile path
        audioFile = self.datalist[index] + ".wav"
        # targetFile path
        targetFile = self.datalist[index] + ".txt"

        # inp: log-mel spectrogram, shape:(inpLen, 80)
        # trgt: target, shape:(trgtLen,) 
        inp, trgt, inpLen, trgtLen = self.prepare_input_logmel(self.dataset, audioFile, targetFile,
                                                            self.charToIx, self.audioParams, self.train)
        return inp, trgt, inpLen, trgtLen

    def __len__(self):
        return len(self.datalist)



    def prepare_input_logmel(self, dataset, audioFile, targetFile, charToIx, audioParams, train):

        if targetFile is not None:

            #reading the target from the target file and converting each character to its corresponding index
            with open(targetFile, "r") as f:
                trgt = f.readline().strip()[7:]

            trgt = [charToIx[char] for char in trgt]
            trgt.append(charToIx["<EOS>"])
            trgt.insert(0, charToIx["<BOS>"])
            trgt = np.array(trgt)
            trgtLen = len(trgt)

        # load file
        sampFreq, inputAudio = wavfile.read(audioFile)
        inputAudio = inputAudio.astype(np.float64)
        # mel spectrogram
        mel = librosa.feature.melspectrogram(y=inputAudio,
                                            sr=sampFreq,
                                            n_mels=audioParams["Dim"],
                                            window = audioParams["Window"],
                                            n_fft=audioParams["WinLen"],
                                            hop_length=audioParams["Shift"]) 

        # log
        log_mel = librosa.power_to_db(mel) # (n_mels, T)
        # normalize
        log_mel = log_mel / np.max(np.abs(log_mel))

        audInp = torch.from_numpy(log_mel) # (80, T)

        
        # SpecAugument
        # I used https://github.com/zcaceres/spec_augment
        if train == True:
            audInp = audInp.unsqueeze(0)
            audInp = time_mask(freq_mask(audInp, F=27, num_masks=2),T=int(audInp.shape[2]*0.05), num_masks=2)
            audInp = audInp.squeeze(0)
        
        audInp = audInp.transpose(0,1) # (T, 80)

        # input Length: The length after applying Conv2dSubsampling.
        inpLen = len(audInp) >> 2
        inpLen -= 1
        inpLen = torch.tensor(inpLen)
        
        if targetFile is not None:
            trgt = torch.from_numpy(trgt)
            trgtLen = torch.tensor(trgtLen)
        else:
            trgt, trgtLen = None, None

        return audInp, trgt, inpLen, trgtLen


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """

    inputBatch = pad_sequence([data[0] for data in dataBatch])
    targetBatch = pad_sequence([data[1] for data in dataBatch])

    inputLenBatch = torch.stack([data[2] for data in dataBatch])
    if not any(data[3] is None for data in dataBatch):
        targetLenBatch = torch.stack([data[3] for data in dataBatch])
    else:
        targetLenBatch = None

    return inputBatch, targetBatch, inputLenBatch, targetLenBatch