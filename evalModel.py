import torch
from torch.utils.data import DataLoader
from model import UNet
from train import Data, DATA_PATH, BATCH_SIZE

MODEL_PATH = "bestModel.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

@torch.no_grad()
def evaluateDice(model, dataLoader):
    model.eval()
    totalDice = 0
    totalSamples = 0

    for inputTensor, targets in dataLoader:
        inputTensor, targets = inputTensor.to(device), targets.to(device)

        sigmoidInputs = torch.sigmoid(model(inputTensor))

        flattenedInputs = sigmoidInputs.view(sigmoidInputs.size(0), sigmoidInputs.size(1), -1)
        flattenedTargets = targets.view(targets.size(0), targets.size(1), -1)

        totalCorrect = (flattenedInputs * flattenedTargets).sum(dim=2)
        smoothing = 0.0001
        diceScore = (2 * totalCorrect + smoothing) / (
            flattenedInputs.sum(dim=2) + flattenedTargets.sum(dim=2) + smoothing
        )

        #diceScore is (batch, channels)
        perSampleDice = diceScore.mean(dim=1)
        totalDice += perSampleDice.sum().item()
        totalSamples += inputTensor.size(0)

    return totalDice / totalSamples

if __name__ == "__main__":
    data = Data(DATA_PATH)
    trainingSplit = int(0.8 * len(data))
    #same seed as train.py so train test split is the same
    _, testingData = torch.utils.data.random_split(
        data, [trainingSplit, len(data) - trainingSplit], generator=torch.Generator().manual_seed(19)
    )
    testingDataLoader = DataLoader(testingData, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet([4, 16, 32, 64, 128, 3]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    finalDiceScore = evaluateDice(model, testingDataLoader)
    print(f"Mean Dice score on test set: {finalDiceScore:.4f}")
    print(f"As a percentage: {finalDiceScore * 100:.2f}%")