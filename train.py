import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

slices = glob.glob("./BraTS2020_TrainingData/BraTS2020_training_data/content/data/*.h5")

for sliceData in slices:
    with h5py.File(sliceData, "r") as file:
        image = file["image"][:]
        mask = file["mask"][:]

    fig, axes = plt.subplots(1, 5, figsize=(14, 7))

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