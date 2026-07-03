import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimisers
from torch.utils.data import Dataset, DataLoader, random_split
from model import UNet

DATA_PATH = "./BraTS2020_TrainingData/BraTS2020_training_data/content/data"
EPOCHS = 25
LEARNING_RATE = 0.001
BATCH_SIZE = 10

class Data(Dataset):
    def __init__(self, dataPath):
        self.filePaths = glob.glob(dataPath + "/*.h5")

    def __len__(self):
        return len(self.filePaths)
    
    def __getitem__(self, i):
        with h5py.File(self.filePaths[i], "r") as file:
            image = file["image"][:].astype(np.float32)
            mask = file["mask"][:].astype(np.float32)
        
        inputTensor = torch.from_numpy(image.transpose(2, 0, 1))
        target = torch.from_numpy(mask.transpose(2, 0, 1))

        return inputTensor, target
    
def costFunction():
    return
    
def trainEpoch(model, optimiser, dataLoader):
    model.train()
    totalCost = 0
    for inputTensor, target in dataLoader:
        optimiser.zero_grad()
        cost = costFunction(model(inputTensor), target)
        cost.backward()
        optimiser.step()
        totalCost += cost.item() * inputTensor.size(0)
    return totalCost / len(dataLoader.dataset)

@torch.no_grad()
def test(model, dataLoader):
    model.eval()
    totalCost = 0
    for inputTensor, target in dataLoader:
        cost = costFunction(model(inputTensor), target)
        totalCost += cost.item() * inputTensor.size(0)
    return totalCost / len(dataLoader.dataset)

if __name__ == "__main__":
    data = Data(DATA_PATH)
    trainingSplit = int(0.8 * len(data))
    trainingData, testingData = random_split(data, [trainingSplit, len(data) - trainingSplit], generator=torch.Generator().manual_seed(19))

    trainingDataLoader = DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testingDataLoader = DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet([4,32,64,128,3])
    optimiser = optimisers.SGD(model.parameters(), lr=LEARNING_RATE)

    for i in range(EPOCHS):
        trainingCost = trainEpoch(model, optimiser, trainingDataLoader)
        testingCost = test(model, testingDataLoader)
        print("Epoch",i,":\ntraining cost =",trainingCost,"\ntesting cost =",testingCost)
        