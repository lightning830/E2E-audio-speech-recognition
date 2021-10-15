import torch
from torch._C import device
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil
from dataset import MyDataset
from dataset import collate_fn
from model import AVSRwithConf2, CustomLoss
from tqdm import tqdm
import sys
from collections import OrderedDict



#character to index mapping
CHAR_TO_INDEX = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
                "A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18,
                "L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14,
                "X":26, "Z":28, "<EOS>":39, "<BOS>":40}    

#index to character reverse mapping
INDEX_TO_CHAR = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
                5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
                11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
                26:"X", 28:"Z", 39:"<EOS>", 40:"<BOS>"}
# zero padding
PAD_IDX = 0
# the number of characters = CHAR_TO_INDEX + zero(padding)
NUM_CLASS = 41
#absolute path to the data directory
DATA_DIRECTORY = 'C:/Users/test/Desktop/python/datasets/mvlrs_v1'
# batch
BATCH_SIZE = 2
# epoch
NUM_STEPS = 50


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt, device):
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return tgt_mask, tgt_padding_mask #(S, S), (Batch, S)


def train(model, trainLoader, optimizer, loss_function, device, trainParams):

    trainingLoss = 0

    with tqdm(trainLoader, leave=False, desc="Train", ncols=75) as pbar:
        for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(pbar):
            
            inputBatch_A, targetBatch = (inputBatch.float()).to(device), (targetBatch.long()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device) # (Batch), (Batch)
            inputBatch_A = inputBatch_A.transpose(0,1) #(Batch, T, 80)

            tgt_mask, tgt_padding_mask = create_mask(targetBatch[:-1,:], device) #(Batch, S, S), (Batch, S)

            targetBatch = targetBatch.transpose(0,1) #(Batch, S)

            optimizer.zero_grad()
            model.train()
            output_Att, output_CTC = model(inputBatch_A, targetBatch[:,:-1], tgt_mask, tgt_padding_mask)

            with torch.backends.cudnn.flags(enabled=False):
                loss, ctc, ce = loss_function(output_Att, output_CTC, targetBatch[:,1:], inputLenBatch, targetLenBatch-1)
                pbar.set_postfix(OrderedDict(CTC=ctc.item(), CE=ce.item()))
            loss.backward()
            optimizer.step()

            trainingLoss = trainingLoss + loss.item()

    trainingLoss = trainingLoss/len(trainLoader)
    return trainingLoss



def evaluate(model, evalLoader, loss_function, device, evalParams):

    evalLoss = 0

    with tqdm(evalLoader, leave=False, desc="Eval", ncols=75) as pbar:

        for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(pbar):
            inputBatch_A, targetBatch = (inputBatch.float()).to(device), (targetBatch.long()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.long()).to(device), (targetLenBatch.long()).to(device) # (Batch,), (Batch,)
            inputBatch_A = inputBatch_A.transpose(0,1) #(Batch, T, 80)
            tgt_mask, tgt_padding_mask = create_mask(targetBatch[:-1,:], device) #(S, S), (Batch, S)
            targetBatch = targetBatch.transpose(0,1) #(Batch, S)

            model.eval()
            with torch.no_grad():
                output_Att, output_CTC = model(inputBatch_A, targetBatch[:,:-1], tgt_mask, tgt_padding_mask)
                with torch.backends.cudnn.flags(enabled=False):
                    loss, ctc, ce = loss_function(output_Att, output_CTC, targetBatch[:,1:], inputLenBatch, targetLenBatch-1)
                    pbar.set_postfix(OrderedDict(CTC=ctc.item(), CE=ce.item()))

            evalLoss = evalLoss + loss.item()

    evalLoss = evalLoss/len(evalLoader)
    return evalLoss


def main():

    matplotlib.use("Agg")

    #seed for random number generators
    np.random.seed(19220297)
    torch.manual_seed(19220297)

    #use gpu
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": 6, "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # declaring the pretrain and the preval datasets and the corresponding dataloaders
        # Window = window to use while computing log-mel spectrogram
        # WinLen = window size 
        # Shift = hop length
        # Dim = log-mel feature's dimention
    audioParams = {"Window":"hann", "WinLen":512, "Shift":160, "Dim":80}
    trainData = MyDataset(["pretrain","train"], DATA_DIRECTORY, CHAR_TO_INDEX, audioParams)
    valData = MyDataset(["val"], DATA_DIRECTORY, CHAR_TO_INDEX, audioParams, train=False)

    trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, **kwargs)
    valLoader = DataLoader(valData, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, **kwargs)
    
    #declaring the model
    model = AVSRwithConf2(NUM_CLASS)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0004, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                     patience=5, threshold=0.001,
                                                     threshold_mode="abs", min_lr=1e-6, verbose=True)
    
    # CTC attention loss: ramda*CTC + (1-ramda)*Attention
    loss_function = CustomLoss(ramda=0.1, beta=0.6)

    # loss curve
    trainingLossCurve = list()
    validationLossCurve = list()

    # Train
    print("/nTraining the model .... /n")

    trainParams = {"spaceIx":CHAR_TO_INDEX[" "], "eosIx":CHAR_TO_INDEX["<EOS>"]}
    valParams = {"decodeScheme":"greedy", "spaceIx":CHAR_TO_INDEX[" "], "eosIx":CHAR_TO_INDEX["<EOS>"]}

    for step in range(NUM_STEPS):

        #train the model for one step
        trainingLoss= train(model, trainLoader, optimizer, loss_function, device, trainParams)
        trainingLossCurve.append(trainingLoss)

        #evaluate the model on validation set
        validationLoss= evaluate(model, valLoader, loss_function, device, valParams)
        validationLossCurve.append(validationLoss)

        #printing the stats after each step
        print("Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f"
              %(step, trainingLoss, validationLoss))

        #make a scheduler step
        scheduler.step(validationLoss)

        #saving the model weights and loss/metric curves in the checkpoints directory
        savePath = "./checkpoints/train-step_{:04d}.pt".format(step)
        torch.save(model.state_dict(), savePath)

        plt.figure()
        plt.title("Loss Curves")
        plt.xlabel("Step No.")
        plt.ylabel("Loss value")
        plt.plot(list(range(1, len(trainingLossCurve)+1)), trainingLossCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationLossCurve)+1)), validationLossCurve, "red", label="Validation")
        plt.legend()
        plt.savefig("./checkpoints/train-step_{:04d}-loss.png".format(step))
        plt.close()

    print("/nTraining Done./n")
    return


if __name__ == "__main__":
    main()