from torchvision import transforms
from torchvision.transforms import ToPILImage
from DataUtils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load model checkpoint
checkpoint = '../Checkpoints/InsulatorInspectorCheckpoint.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint, map_location = device)
epoch = checkpoint['epoch'] + 1
print('Loaded checkpoint from epoch', epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

#Image transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) #ImageNet mean and std

labelMap = {'No issues': 0, 'notbroken-notflashed':0, 'Insulator':1, 'Flashover damage':2, 'Broken':3}
labelColorMap = {'No issues':'green', 'Insulator':'blue', 'Flashover damage':'yellow', 'Broken':'red'}
revLabelMap = {0:'No issues', 1:'Insulator', 2:'Flashover damage', 3:'Broken'}

def detect(original_image, minScore, maxOverlap, topK, suppress = None):
    """
    Detect objects in image with trained SSD300, visualize results.

    :param image: PIL image
    :param minScore: minimum score for detection of certain class
    :param maxOverlap: maximum overlap two boxes can have so before the one with the lower score is suppressed via NMS 
    :param topK: number of top scoring boxes to keep before NMS
    :param suppress: classes to suppress, if None all classes are considered
    """
    
    image = normalize(to_tensor(resize(original_image))) #Transform image, convert to tensor

    image = image.to(device)    #Move to default device

    predictedBoxes, predictedScores = model(image.unsqueeze(0)) #Predict boxes and scores, insert batch dimension

    detectedBoxes, detectedLabels, detectedScores = model.detectObjects(predictedBoxes, predictedScores, minScore = minScore,
                                                                        maxOverlap = maxOverlap, topK = topK)   #Detect objects
    
    detectedBoxes = detectedBoxes[0].to('cpu')  #Move to cpu

    #Transform image to original dimensions. Width and height repeated for four corners of box
    originalDims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)

    detectedBoxes = detectedBoxes * originalDims    #Scale boxes to original dimensions

    detectedLabels = [revLabelMap[l] for l in detectedLabels[0].to('cpu').tolist()]     #Convert labels to string names

    #If no issues detected (this is the background class, represented with 0), just return image
    if detectedLabels == ['No issues']:
        return image
    
    #Annotate the image

    annotatedImage = original_image

    draw = ImageDraw.Draw(annotatedImage)
    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15) #For Mac, need to specify font path

    #Suppress specific classes if needed
    for i in range(detectedBoxes.size(0)):
        if suppress is not None:
            if detectedLabels[i] in suppress:
                continue
        
        #Draw boxes
        boxLocation = detectedBoxes[i].tolist()
        draw.rectangle(xy = boxLocation, outline = labelColorMap[detectedLabels[i]]) #Draw rectangle
        draw.rectangle(xy = [l + 1. for l in boxLocation], outline = labelColorMap[detectedLabels[i]]) #Draw second rectangle 1 pixel apart for more thickness

        #Text
        textSize = font.getbbox(detectedLabels[i].upper())
        textLocation = [boxLocation[0] + 2., boxLocation[1] - textSize[1]]
        textBoxLocation = [boxLocation[0], boxLocation[1] - textSize[1], boxLocation[0] + textSize[0] + 4., boxLocation[1]]
        draw.rectangle(xy = textBoxLocation, fill = labelColorMap[detectedLabels[i]])
        draw.text(xy = textLocation, text = detectedLabels[i].upper(), fill = 'white', font = font)

    del draw #Free up memory
    print(annotatedImage.height, annotatedImage.width)
    return annotatedImage

def main():
    test_images_path = '../Data/ProcessedData/test_Images.json'
    with open(test_images_path, 'r') as f:
        test_images = json.load(f)
    i = 0
    imgPath = test_images[i]
    img = Image.open(imgPath, mode = 'r')
    img = img.convert('RGB')
    detect(img, minScore = 0.2, maxOverlap = 0.5, topK = 100).show()
    
if __name__ == '__main__':
    main()