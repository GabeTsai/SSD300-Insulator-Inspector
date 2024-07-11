import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt 
from Model import SSD300, MultiBoxLoss
from Datasets import InsulatorDataset
from DataUtils import *

dataFolderPath = '/Users/HP/Documents/GitHub/SSD300-Insulator-Inspector/Data/ProcessedData'

#Model params
numClasses = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)

#Training params
checkpointPath = None
batchSize = 8
iterations = 120000
workers = 4 #Number of workers for dataloader
lr = 3e-4
# lrDecayFraction = 0.1
# momentum = 0.9
# weightDecay = 5e-4
# gradClip = None

cudnn.benchmark = True

def main():
    """
    Train the SSD300 model
    """
    # decayLrAt = [80000, 100000]

    #Initialize model
    if checkpointPath is None:
        startEpoch = 0
        model = SSD300(numClasses=numClasses)
        #Initialize optimizer, twice the default lr for biases, per Caffe repo. Why? Makes the stuff converge faster.
        biases = []
        weights = []
        for paramName, param in model.named_parameters():
            if param.requires_grad:
                if paramName.endswith('.bias'):
                    biases.append(param)
                else:
                    weights.append(param)
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': weights}],
        #                             lr=lr, momentum=momentum, weight_decay=weightDecay)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, weight_decay = 1e-5)
    #Alternatively, load a checkpoint if it exists
    else:
        checkpoint = torch.load(checkpointPath)
        startEpoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % startEpoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    model = model.to(device)
    criterion = MultiBoxLoss(priorsCenter=model.priorsCenter).to(device)

    trainDataset = InsulatorDataset(dataFolderPath, 'train')
    valDataset = InsulatorDataset(dataFolderPath, 'val')

    trainLoader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=trainDataset.collateFunc,
        num_workers=workers,
        pin_memory=True
    )

    valLoader = torch.utils.data.DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False, #no need to shuffle validation dataset
        collate_fn=valDataset.collateFunc,
        num_workers=workers,
        pin_memory=True
    )

    epochs = iterations // (len(trainDataset) // batchSize)
    # decayLrAt = [it // (len(trainDataset) // batchSize) for it in decayLrAt]

    trainLossList = []
    epochList = []
    valLossList = []

    for epoch in range(startEpoch, epochs):
        # if epoch in decayLrAt:
        #     decayLearningRate(optimizer, lrDecayFraction)

        loss = train(trainLoader = trainLoader, model = model, 
              criterion = criterion, optimizer = optimizer, epoch = epoch)

        valLoss = validate(valLoader = valLoader, model = model, 
              criterion = criterion, epoch = epoch, optimizer = optimizer)

        epochList.append(epoch)
        trainLossList.append(loss)
        valLossList.append(valLoss)
    plotLosses(trainLossList, valLossList, epochList)

def train(trainLoader, model, criterion, optimizer, epoch):
    """
    One epoch of training.

    :param trainLoader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()

    batchTime = AverageMeter()
    dataLoadingTime = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    epochLossSum = 0
    for i, (images, bBoxes, labels) in enumerate(trainLoader):
        images = images.to(device)
        bBoxes = [b.to(device) for b in bBoxes]
        labels = [l.to(device) for l in labels]

        #Forward pass
        predictedLocs, predictedScores = model(images)
        
        #Compute multibox loss
        loss = criterion(predictedLocs, predictedScores, bBoxes, labels)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # if gradClip is not None:
        #     clipGradient(optimizer, gradClip)
        
        #Update model params
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batchTime.update(time.time() - start)

        start = time.time() 
        if i % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batchTime.val:.3f} ({batchTime.avg:.3f})\t'
                  'Data {dataLoadingTime.val:.3f} ({dataLoadingTime.avg:.3f})\t'
                  'Train Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(trainLoader), batchTime=batchTime,
                   dataLoadingTime=dataLoadingTime, loss=losses))
        epochLossSum += loss.item()
    del predictedLocs, predictedScores, images, bBoxes, labels #Free up memory 
    return epochLossSum/len(trainLoader) #averaged loss after one epoch of training

def validate(valLoader, model, criterion, epoch, optimizer):
    """
    One epoch of validation

    :param valLoader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :param epoch: epoch number
    """

    model.eval()

    batchTime = AverageMeter()
    dataLoadingTime = AverageMeter()
    losses = AverageMeter()

    epochLossSum = 0
    start = time.time()
    lowestValLoss = np.inf

    for i, (images, bBoxes, labels) in enumerate(valLoader):
        images = images.to(device)
        bBoxes = [b.to(device) for b in bBoxes]
        labels = [l.to(device) for l in labels]

        #Forward pass
        predictedLocs, predictedScores = model(images)
        
        #Compute multibox loss
        loss = criterion(predictedLocs, predictedScores, bBoxes, labels)
        if loss < lowestValLoss:
            lowestValLoss = loss
            print("Lower validation loss reached, saving checkpoint")
            saveCheckpoint(epoch, model, optimizer, '../Checkpoints')
        losses.update(loss.item(), images.size(0))
        batchTime.update(time.time() - start)

        start = time.time() 
        epochLossSum += loss.item()
        if i % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batchTime.val:.3f} ({batchTime.avg:.3f})\t'
                  'Data {dataLoadingTime.val:.3f} ({dataLoadingTime.avg:.3f})\t'
                  'Validation Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(valLoader), batchTime=batchTime,
                   dataLoadingTime=dataLoadingTime, loss=losses))
            
    del predictedLocs, predictedScores, images, bBoxes, labels #Free up memory 
    print('Averaged Validation Loss: ', epochLossSum/len(valLoader))
    return epochLossSum/len(valLoader) #averaged loss after one epoch of validation

def plotLosses(trainLosses, valLosses, epochs):
    """
    Plot the losses vs epochs

    :param losses: list of losses
    :param epochs: list of epochs
    """
    plt.plot(epochs, trainLosses, label = 'Train Loss')
    plt.plot(epochs, valLosses, label = 'Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Losses.png')

if __name__ == '__main__':
    main()