import json
import os
import torch 
import random
import torchvision.transforms.functional as FT
from PIL import Image, ImageStat, ImageDraw, ImageFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labelMap = {'Background':0, 'Insulator':1, 'Flashover damage':2, 'Broken':3, 'No issues':4, 'notbroken-notflashed':4}

def coordToBoundary(bboxes):
    """
    Convert coordinates to boundary form
    :param bboxes: n by 4 tensor of bboxes
    :return: bboxes in boundary format (x_min, y_min, x_max, y_max)
    """
    for i in range(len(bboxes)):
        row = bboxes[i]
        if (row[0] == 0):
            row[2] -= 1
        elif (row[1] == 0):
            row[3] -= 1
        row[2] += row[0]
        row[3] += row[1]
    return bboxes

def boundaryToCenter(bboxes):
    """
    Convert boundary coordinates to center form
    :param bboxes: n by 4 tensor of boundary bboxes
    :return: bboxes in center format (cx, cy, w, h)
    """
    return torch.cat([(bboxes[:, 2:] + bboxes[:, 2:]) / 2, bboxes[:, 2:] - bboxes[:, :2]], 1)
    
def centerToBoundary(bboxes):
    """
    Convert center coordinates to boundary form (x_min, y_min, x_max, y_max)

    :param bboxes: n by 4 tensor of center-size bboxes
    :return: bboxes in boundary format, tensor (n_boxes, 4)
    """
    return torch.cat([bboxes[:, :2] - (bboxes[:, 2:] / 2), bboxes[:, :2] + (bboxes[:, 2:] / 2)], 1)

def encodeOffsets(centerBboxes, centerPriors):
    """
    Encode a bounding box in center form with respect to a prior box in center form.

    gcx = (cx - pcx) / pw
    gcy = (cy - pcy) / ph
    gw = log(w / pw)
    gh = log(h / ph)

    :param centerBboxes: tensor of bounding boxes in center form (n_priors, 4)
    :param centerPriors: tensor of prior boxes in center form with respect to which encoding must be performed (n_priors, 4)
    :return: encoded offsets with respect to centerPriors (n_priors, 4)

    #10 and 5 are chosen as normalization factors to approximate a gaussian distribution for adjusting prior box or 
    #scaling localization gradient
    """

    return torch.cat([(centerBboxes[:, :2] - centerPriors[:, :2]) / (centerPriors[:, 2:] / 10), 
                      torch.log(centerBboxes[:, 2:] / centerPriors[:, 2:]) * 5], 1)

def decodeOffsets(predictedOffsets, centerPriors):
    """
    Decode model's bounding box offset predictions into center-size coordinates. Inverse of function above.
    
    :param predictedOffsets: encoded bounding boxes, or the output of the model (n_priors, 4)
    :param centerPriors: tensor of prior boxes in center form with respect to which encoding must be performed (n_priors, 4)
    :return: encoded offsets with respect to centerPriors (n_priors, 4)
    """

    return torch.cat([predictedOffsets[:, :2] * centerPriors[:, 2:] / 10 + centerPriors[:, :2], #cx, cy
                      torch.exp(predictedOffsets[:, 2:] / 5) * centerPriors[:, 2:]], 1) #w, h

def findIntersection(bboxes1, bboxes2):
    """
    Find intersection of every box pairing between two sets(tensors) of bboxes in boundary coords

    :param bboxes1: tensor of bboxes in boundary form (n_bboxes1, 4)
    :param bboxes2: tensor of bboxes in boundary form (n_bboxes2, 4)
    :return: tensor of intersections of all bboxes in set 1 with respect to bboxes in set 2 tensor of dims (n1, n2)
    """

    lowerBounds = torch.max(bboxes1[:, :2].unsqueeze(1), bboxes2[:, :2].unsqueeze(0)) #(n1, n2, 2)
    upperBounds = torch.min(bboxes1[:, 2:].unsqueeze(1), bboxes2[:, 2:].unsqueeze(0)) #(n1, n2, 2)

    intersection = torch.clamp(upperBounds - lowerBounds, min = 0) #(n1, n2, 2)
    #Calculate intersection area
    return intersection[:, :, 0] * intersection[:, :, 1] #(n1, n2)

def findJaccardOverlap(bboxes1, bboxes2):
    """
    Find Jaccard Overlap, or Intersection over Union (IoU) of two sets of bboxes in boundary coordinates. 

    :param bboxes1: tensor of bboxes in boundary form (n_bboxes1, 4)
    :param bboxes2: tensor of bboxes in boundary form (n_bboxes2, 4)
    :return: Jaccard Overlap of all bboxes in set 1 with respect to bboxes in set 2 tensor of dims (n1, n2)
    """

    #Find intersections
    intersection = findIntersection(bboxes1, bboxes2)

    #Find areas of each box in both sets
    areas1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1]) #(n1)
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1]) #(n2)

    #Find union of all bboxes in set 1 with respect to bboxes in set 2
    union = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersection #(n1, n2)

    return intersection/union #(n1, n2), IoU of all bboxes in set 1 with respect to bboxes in set 2
    
def decimate(tensor, m):
    """
    Downsample a tensor, keeping every m'th value, used when converting FC to Conv layers of a smaller size.
    :param tensor: input tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor
    
    """
    assert tensor.dim() == len(m)

    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim = d, index = torch.arange(start = 0, end = tensor.size(d), step = m[d]).long())

    return tensor

def horizontalFlip(image, bboxes):
    """
    Horizontally flip image and bboxes
    :param image: PIL image, bboxes in boundary form
    :return: horizontally flipped image and bboxes
    """
    flippedImage = FT.hflip(image)

    flippedBboxes = bboxes
    flippedBboxes[:, 0] = image.width - bboxes[:, 0] - 1
    flippedBboxes[:, 2] = image.width - bboxes[:, 2] - 1

    flippedBboxes = flippedBboxes[:, [2, 1, 0, 3]]

    return flippedImage, flippedBboxes

def resize(image, bboxes):
    """
    Resize image to 300 x 300 for SSD 300. Return resized image and bboxes in fractional form.
    :param image: PIL image
    :param bboxes: bboxes in boundary form
    :return: resized image, fractional bboxes
    """
    dims = (300, 300)
    bboxes = coordToBoundary(bboxes)

    newImage = FT.resize(image, dims)
    prevDims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)

    newBboxes = bboxes/prevDims #Convert to fractional (percentage) coordinates

    return newImage, newBboxes

def photometricDistort(image):
    """
    # Apply brightness, contrast, saturation, and hue distortions, each with 50 % chance in random order
    :param image: PIL image
    :return: distorted PIL image
    """
    newImage = image

    distortions = [FT.adjust_brightness, FT.adjust_contrast, FT.adjust_saturation, FT.adjust_hue]
    random.shuffle(distortions)

    #Values based on caffe implementation
    for d in distortions:
        if random.random() > 0.5:
            if d.__name__ == 'adjust_hue':
                #change hue by random amount, normalized by max hue value (255) since PyTorch needs normalized values
                adjust_factor = random.uniform(-18/255., 18/255.) 
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            #Apply d 
            newImage = d(newImage, adjust_factor)
    return newImage

def transform(image, bboxes, labels, split):
    assert split in {'train', 'test', 'val'}

    #Mean and SD of ImageNet data for base trained VGG16 model from torchvision
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    newImage = image
    newBboxes = bboxes
    newLabels = labels

    if (split == 'train'):
        #Randomly apply photometric distortions
        newImage = photometricDistort(newImage)

        #Randomly apply horizontal flip
        if random.random() > 0.5:
            newImage, newBboxes = horizontalFlip(newImage, newBboxes)

    #Resize image to 300 x 300 for SSD 300
    newImage, newBboxes = resize(newImage, newBboxes)

    #Convert PIL image to tensor
    newImage = FT.to_tensor(newImage)

    #Normalize image to ImageNet values that base VGG trained on
    newImage = FT.normalize(newImage, mean = mean, std = std)

    return newImage, newBboxes, newLabels

def createBackgroundImages(imagePathList, imageDir, numImages, objectList):
    """
    Generates background class from 10% of images in the dataset by replacing the bounding boxes of the insulators with
    gaussian blur.
    :param imagePathList: list of image paths
    :param imageDir: directory for background images to be stored
    :param numImages: number of background images to generate, expressed as a fraction of total images
    :param objectList: list of dictionaries with bounding boxes ('bbox') and labels ('labels') for each image
    """
    totalImages = len(imagePathList)
    numBackgroundImages = int(totalImages * numImages)  # Calculate number based on fraction provided

    # Sample a subset of image paths for background generation
    imagePathIndexList = random.sample(range(totalImages), numBackgroundImages)

    for count, idx in enumerate(imagePathIndexList, start=1):
        imagePath = imagePathList[idx]
        imageFullPath = os.path.join(imageDir, imagePath)
        image = Image.open(imageFullPath).convert('RGB')
        objects = objectList[idx]  # Assuming objectList is correctly aligned with imagePathList

        for bbox, label in zip(objects['bbox'], objects['labels']):
            if label == 1:  # Assuming 1 signifies insulator
                # Create a mask for the current object to blur
                mask = Image.new("L", image.size, 0)
                draw = ImageDraw.Draw(mask)
                # Draw a rectangle on the mask where the object is, filling it with white (255)
                draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], fill=255)
                # Blur the entire image
                blurred_image = image.filter(ImageFilter.GaussianBlur(70))
                # Paste the blurred object back into the image using the mask
                image.paste(blurred_image, mask=mask)

        savePath = os.path.join(imageDir, f'bg{count}.jpg')
        image.save(savePath)
        objectList.append({'bbox': [], 'labels': [0]})  # Add background label to object list
        imagePathList.append(savePath)
        print(f'Background image {count} saved to {savePath}')

def test(i):
    """
    Test that transform works
    :param: i: index of image to test from json file
    """
    with open('/Users/HP/Documents/InsulatorInspector/Data/ProcessedData/test_Images.json', 'r') as f:
        images = json.load(f)
    with open('/Users/HP/Documents/InsulatorInspector/Data/ProcessedData/test_Objects.json', 'r') as f:
        objects = json.load(f)
    print(images[i])
    image = Image.open(images[i], mode = 'r')
    image = image.convert('RGB')
    object = objects[i]
    print(object)
    bboxes = torch.FloatTensor(object['bbox'])
    newImage, newBboxes, newLabels = transform(image, bboxes, object['labels'], 'test')
    newImage = FT.to_pil_image(newImage)
    newImage.show()
    print(newBboxes)
    print(newLabels)

def calculatemAP(detectedBoxes, detectedLabels, detectedScores, trueBoxes, trueLabels, trueDifficulties):
    """
    Calculate Mean Average Precision (mAP) of detected objects.

    :param detectedBoxes: list of tensors of detected bounding boxes in boundary form, one tensor for each image holding detected object bboxes 
    :param detectedLabels: list of tensors of detected labels, one tensor for each image holding detected object labels 
    :param detectedScores: list of tensors of detected scores, one tensor for each image holding detected object scores 
    :param trueBoxes: list of tensors of true bounding boxes in boundary form, one tensor for each image holding true object bboxes
    :param trueLabels: list of tensors of true labels, one tensor for each image holding true object labels
    :param trueDifficulties: list of tensors of true difficulties, one tensor for each image holding true object difficulties (0 or 1 for each object)
    :return: list of average precisions for all classes, mAP
    """

    assert len(detectedBoxes) == len(detectedLabels) == len(detectedScores) == len(trueBoxes) == len(trueLabels)
    #These should all equal the same (number of images)

    numClasses = len(labelMap)

    #Store all ground truth boxes and labels (objects) in a continuous tensor and keep track of image they're from
    trueImages = []
    for i in range(len(trueLabels)):
        trueImages += [i] * trueLabels[i].size(0) #each element is an image index repeated for each object in the image
    trueImages = torch.LongTensor(trueImages) #(numObjects), total number of objects across all images
    trueBoxes = torch.cat(trueBoxes, dim = 0) #(numObjects, 4)
    trueLabels = torch.cat(trueLabels, dim = 0) #(numObjects)
    trueDifficulties = torch.cat(trueDifficulties, dim = 0) #(numObjects)

    assert trueImages.size(0) == trueBoxes.size(0) == trueLabels.size(0) == trueDifficulties.size(0)

    #Store all detections/predictions in a single continuous tensor, keeping track of the images they're from
    detectedImages = []
    for i in range(len(detectedLabels)):
        detectedImages += [i] * detectedLabels[i].size(0)
    detectedImages = torch.LongTensor(detectedImages) #(numDetections)
    detectedBoxes = torch.cat(detectedBoxes, dim = 0) #(numDetections, 4)
    detectedLabels = torch.cat(detectedLabels, dim = 0) #(numDetections)
    detectedScores = torch.cat(detectedScores, dim = 0) #(numDetections)

    assert detectedImages.size(0) == detectedBoxes.size(0) == detectedLabels.size(0) == detectedScores.size(0)

    #Calculate APs for each class

    averagePrecisions = torch.zeros((numClasses - 1), dtype = torch.float) #Exclude background class
    for c in range(1, numClasses):
        #Extract all true detections and ground truth objects for class c
        trueClassImages = trueImages[trueLabels == c] #(num true class objects), images which objects belong to
        trueClassBoxes = trueBoxes[trueLabels == c] #(num true class objects, 4)
        trueClassDifficulties = trueDifficulties[trueLabels == c] #(num true class objects)
        numEasyClassObjects = (1 - trueClassDifficulties).sum().item() #Number of non-difficult objects

        #Track which true objects belonging to this class have been 'detected'. As of this point, none.
        trueClassBoxesDetected = torch.zeros((trueClassBoxes.size(0)), dtype = torch.uint8).to(device) #0 for not detected, 1 for detected

        #Get all detections for this class
        detectedClassImages = detectedImages[detectedLabels == c] #(num detected class objects)
        detectedClassBoxes = detectedBoxes[detectedLabels == c] #(num detected class objects, 4)
        detectedClassScores = detectedScores[detectedLabels == c] #(num detected class objects)
        numDetectedClassObjects = detectedClassBoxes.size(0)
        if numDetectedClassObjects == 0: continue
        
        #Sort detections by decreasing order of scores
        detectedClassScores, sortIndices = torch.sort(detectedClassScores, dim = 0, descending = True) #(num detected class objects)
        detectedClassBoxes = detectedClassBoxes[sortIndices] #(num detected class objects, 4)
        detectedClassImages = detectedClassImages[sortIndices] #(num detected class objects)

        truePositives = torch.zeros((numDetectedClassObjects), dtype = torch.float).to(device) #(num detected class objects)
        falsePositives = torch.zeros((numDetectedClassObjects), dtype = torch.float).to(device) #(num detected class objects)

        for d in range(numDetectedClassObjects):
            detectedBox = detectedClassBoxes[d].unsqueeze(0) #(1, 4)
            image = detectedClassImages[d] #Image index for detected object

            #Find objects in image with same class, their difficulties and whether they've been detected
            objectBoxes = trueClassBoxes[trueClassImages == image] #(num true class objects in image, 4)
            objectDifficulties = trueClassDifficulties[trueClassImages == image] #(num true class objects in image)
            if objectBoxes.size(0) == 0: #no objects of this class in image (hence no boxes)
                falsePositives[d] = 1
                continue
            
            #Calculate Jaccard Overlap between detected box and all true boxes in image
            overlaps = findJaccardOverlap(detectedBox, objectBoxes) #(1, num true class objects in image)
            maxOverlap, maxIndex = overlaps.max(dim = 1) #scalars

            originalIndex = torch.LongTensor(range(trueClassBoxes.size(0)))[trueClassImages == image][maxIndex] #index of max overlap object in trueClassBoxes

            if maxOverlap.item() > 0.5:
                if objectDifficulties[maxIndex] == 0: #ignore "difficult objects"
                    #If object hasn't already been detected -> true positive
                    if trueClassBoxesDetected[originalIndex] == 0:
                        truePositives[d] = 1
                        trueClassBoxesDetected[originalIndex] = 1
                    else: #Object has already been detected -> false positive (multiple detections)
                        falsePositives[d] = 1
            #If overlap is less than 0.5, it's a false positive (poor detection)
            else:
                falsePositives[d] = 1

        #Calculate cumulative precision and recall
        cumTruePositives = torch.cumsum(truePositives, dim = 0) #(num detected class objects)
        cumFalsePositives = torch.cumsum(falsePositives, dim = 0) #(num detected class objects)
        cumPrecision = cumTruePositives / (cumTruePositives + cumFalsePositives + 1e-10) #(num detected class objects)
        cumRecall = cumTruePositives / numEasyClassObjects #(num detected class objects)

        #Find mean of max precision at each recall level above threshold 't'
        recallThresholds = torch.arange(start = 0, end = 1.1, step = 0.1).tolist() #(11) recall thresholds
        precisions = torch.zeros((len(recallThresholds)), dtype = torch.float).to(device) #(11) for each recall threshold
        for i, t in enumerate(recallThresholds):
            recallsAboveThreshold = cumRecall >= t
            if recallsAboveThreshold.any():
                precisions[i] = cumPrecision[recallsAboveThreshold].max()
            else:
                precisions[i] = 0
        averagePrecisions[c - 1] = precisions.mean() #c-1 since we excluded the background class

    #Calculate mAP (finally)
    mAP = averagePrecisions.mean().item()

    #store class-wise average precisions and mAP in a dictionary
    averagePrecisions = {labelMap[c+1]: v for c, v in enumerate(averagePrecisions.tolist())}

    return averagePrecisions, mAP

def decayLearningRate(optimizer, scale):
    """
    Decay learning rate by a factor of scale
    :param optimizer: optimizer object
    :param scale: factor by which to decay learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale
    print(f"Learning rate decayed by a factor of {scale} to {optimizer.param_groups[1]['lr']:.6f}")
                    
def clipGradient(optimizer, gradClip):
    """
    Clip gradients to val between -gradClip and gradClip during backprop to prevent exploding gradients
    :param optimizer: optimizer object
    :param gradClip: value to clip gradients to
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-gradClip, gradClip)

def saveCheckpoint(epoch, model, optimizer):
    """
    Save model checkpoint

    :param epoch: epoch number
    :param model: model object
    :param optimizer: optimizer object
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    filename = 'insulatorInspectorCheckpoint.pth.tar'
    torch.save(state, filename)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    return 0

if __name__ == '__main__':
    main()