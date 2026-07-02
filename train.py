import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

DATA_PATH = "./BraTS2020_TrainingData/BraTS2020_training_data/content/data"

EPOCHS = 25
LEARNING_RATE = 0.001

class Data(Dataset):
    def __init__(self, dataPath):
        self.filePaths = glob.glob(dataPath + "/*.h5")

    def __len__(self):
        return len(self.filePaths)
    
    def __getitem__(self, i):
        with h5py.File(self.filePaths[i], "r") as file:
            image = file["image"][:].astype(np.float32)
            mask = file["mask"][:].astype(np.float32)
        
        inputTensor = torch.from_numpy(image)
        target = torch.from_numpy(mask)

        return inputTensor, target


if 1 == 1:
    quit()

#slices = glob.glob("./BraTS2020_TrainingData/BraTS2020_training_data/content/data/*.h5")
slices = glob.glob("./test/*.h5")

for slicePath in sorted(slices):
    with h5py.File(slicePath, "r") as file:
        image = file["image"][:]
        mask = file["mask"][:]

    fig, axes = plt.subplots(1, 5, figsize=(14, 7))

    fig.suptitle(slicePath.split("\\")[-1])

    colours = np.array([[1,0,0],[0,1,0],[0,0.5,1]], dtype=np.float32)

    for y in range(4):
        img = image[..., y].astype(np.float32)

        axes[y].imshow(img)

    rgb = np.zeros((240, 240, 3), dtype=np.float32)
    colours = np.array([[1,0,0],[0,1,0],[0,0,1]])
    for i in range(3):
        channelMask = mask[..., i].astype(bool)
        rgb[channelMask] = colours[i]

    axes[4].imshow(rgb)

    plt.tight_layout()
    plt.pause(1)
    plt.close()