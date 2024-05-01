from torch import nn 
from torch.nn import functional as F
import torchvision
from DataTransforms import *
from math import sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG16Base(nn.Module):
    """
    VGG16 base model
    """
    def __init__(self):
        super(VGG16Base, self).__init__()

        #VGG16 convolutional layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)    
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode = True) 

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding = 1) #stride and padding are 1, so size remains the same

        #Dilate the kernel by a factor of six to fill in holes from decimation
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.loadPretrainedLayers()
    
    def forward(self, image):
        """
        Forward pass through the network

        :param image: images, tensor of dimensions (N, 3, 300, 300), where N is batch size
        :return: lower-level activation maps of conv4_3 and conv7
        """
        out = F.relu(self.conv1_1(image)) #(N, 64, 300, 300)
        out = F.relu(self.conv1_2(out)) #(N, 64, 300, 300)
        out = self.pool1(out) #(N, 64, 150, 150)

        out = F.relu(self.conv2_1(out)) #(N, 128, 150, 150)
        out = F.relu(self.conv2_2(out)) #(N, 128, 150, 150)
        out = self.pool2(out) #(N, 128, 75, 75)

        out = F.relu(self.conv3_1(out)) #(N, 256, 75, 75)
        out = F.relu(self.conv3_2(out)) #(N, 256, 75, 75)
        out = F.relu(self.conv3_3(out)) #(N, 256, 75, 75)
        out = self.pool3(out) #(N, 256, 38, 38), ceil_mode = True added 1 to 37 to make it 38

        out = F.relu(self.conv4_1(out)) #(N, 512, 38, 38)
        out = F.relu(self.conv4_2(out)) #(N, 512, 38, 38)
        out = F.relu(self.conv4_3(out)) #(N, 512, 38, 38)
        conv4_3_features = out #(N, 512, 38, 38)
        out = self.pool4(out) #(N, 512, 19, 19)

        out = F.relu(self.conv5_1(out)) #(N, 512, 19, 19)
        out = F.relu(self.conv5_2(out)) #(N, 512, 19, 19)
        out = F.relu(self.conv5_3(out)) #(N, 512, 19, 19)
        out = self.pool5(out) #(N, 512, 19, 19) stride = 1, padding = 1, so size remains the same

        out = F.relu(self.conv6(out)) #(N, 1024, 19, 19)
        conv7_features = F.relu(self.conv7(out)) #(N, 1024, 19, 19)

        #Return lower-level feature maps
        return conv4_3_features, conv7_features

    def loadPretrainedLayers(self):
        #Get the names of the parameters in the model
        stateDict = self.state_dict()
        parameterNames = list(stateDict.keys())
        #Load in pretrained VGG16 model
        pretrainedStateDict = torchvision.models.vgg16(weights=True).state_dict()
        pretrainedParameterNames = list(pretrainedStateDict.keys())

        for i, parameter in enumerate(parameterNames[:-4]): #exclude conv6 and conv7 weights and biases
            stateDict[parameter] = pretrainedStateDict[pretrainedParameterNames[i]]

        #Convert fc6, fc7 to convolutional layers, then decimate to sizes of conv6 and conv7
        #An FC layer that takes in C*H*W flattened 2D input is the same as a Conv layer with kernel (H,W), 
        #input channels C, padding = 0, output channels K operating on the same (C * H * W) input 2D image
            
        fc6ToConvWeight = pretrainedStateDict['classifier.0.weight'].view(4096, 512, 7, 7) #fc6 has flattened input size of 7 * 7 *512, output of 4096
        fc6ToConvBias = pretrainedStateDict['classifier.0.bias'] #4096 biases
        stateDict['conv6.weight'] = decimate(fc6ToConvWeight, m=[4, None, 3, 3]) #decimate by 4 in height and width, keep all channels --> (1024, 512, 3, 3)
        stateDict['conv6.bias'] = decimate(fc6ToConvBias, m=[4]) #decimate by 4 --> (1024)

        fc7ToConvWeight = pretrainedStateDict['classifier.3.weight'].view(4096, 4096, 1, 1) #fc7 has flattened input size of 4096, output of 4096 by 4096
        fc7ToConvBias = pretrainedStateDict['classifier.3.bias'] #4096 biases
        stateDict['conv7.weight'] = decimate(fc7ToConvWeight, m=[4, 4, None, None]) #decimate by 4 in height and width, keep all channels --> (1024, 1024, 1, 1)
        stateDict['conv7.bias'] = decimate(fc7ToConvBias, m=[4]) #decimate by 4 --> (1024)

        self.load_state_dict(stateDict)
        
        print("\nLoaded pretrained VGG16 weights and biases successfully.\n")

class AuxiliaryConvolutions(nn.Module):
    """
    Auxiliary convolutions for SSD300. These provide higher-level feature maps (objects/shapes)
    Stride 2 in every second layer reduces input dimensions to subsequent layer.
    """
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0) #kernel_size = 1, padding = 0, so size remains the same - output: 19x19
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) #output: 10x10

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size = 1, padding = 0) 
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1) #output: 5x5

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size = 1, padding = 0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size = 3, padding = 0) #output: 3x3

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size = 1, padding = 0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size = 3, padding = 0) #output: 1x1

        self.initAuxiliaryConvolutions()

    def initAuxiliaryConvolutions(self):
        """
        Initialize weights and biases with xavier uniform and zeros respectively
        """
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, conv7_features):
        """
        :param conv7_features: lower-level conv7 feature map, tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps from conv8_2, conv9_2, conv10_2, conv11_2
        """
        
        out = F.relu(self.conv8_1(conv7_features)) #(N, 256, 19, 19)
        out = F.relu(self.conv8_2(out)) #(N, 512, 10, 10)
        conv8_2_features = out #(N, 512, 10, 10)
        
        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_features = out #(N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_features = out #(N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))
        conv11_2_features = F.relu(self.conv11_2(out)) #(N, 256, 1, 1)

        return conv8_2_features, conv9_2_features, conv10_2_features, conv11_2_features

class PredictionConvolutions(nn.Module):
    """
    Convolutional layers used to predict class scores and bounding boxes.

    Bounding box locations are predicted as four offsets for each of the 8732 prior boxes.
    Predicted boxes are represented as the offsets from their respective priors.

    Class scores represent score/probability of each the four object classes in each of the 8732 located bounding boxes.
    """

    def __init__(self, numClasses):
        super(PredictionConvolutions, self).__init__()
        self.numClasses = numClasses

        #Number of prior boxes per position in each activation/feature map

        numPriorBoxes = {'conv4_3': 4,
                         'conv7': 6,
                        'conv8_2': 6,
                        'conv9_2': 6,
                        'conv10_2': 4,
                        'conv11_2': 4}

        #Convolutions to predict localization offsets (4 offsets per prior box) and class scores (numClasses per prior box)
        self.offset_conv4_3 = nn.Conv2d(512, numPriorBoxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.offset_conv7 = nn.Conv2d(1024, numPriorBoxes['conv7'] * 4, kernel_size=3, padding=1)
        self.offset_conv8_2 = nn.Conv2d(512, numPriorBoxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.offset_conv9_2 = nn.Conv2d(256, numPriorBoxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.offset_conv10_2 = nn.Conv2d(256, numPriorBoxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.offset_conv11_2 = nn.Conv2d(256, numPriorBoxes['conv11_2'] * 4, kernel_size=3, padding=1)

        self.class_conv4_3 = nn.Conv2d(512, numPriorBoxes['conv4_3'] * numClasses, kernel_size=3, padding=1)
        self.class_conv7 = nn.Conv2d(1024, numPriorBoxes['conv7'] * numClasses, kernel_size=3, padding=1)
        self.class_conv8_2 = nn.Conv2d(512, numPriorBoxes['conv8_2'] * numClasses, kernel_size=3, padding=1)
        self.class_conv9_2 = nn.Conv2d(256, numPriorBoxes['conv9_2'] * numClasses, kernel_size=3, padding=1)
        self.class_conv10_2 = nn.Conv2d(256, numPriorBoxes['conv10_2'] * numClasses, kernel_size=3, padding=1)
        self.class_conv11_2 = nn.Conv2d(256, numPriorBoxes['conv11_2'] * numClasses, kernel_size=3, padding=1)

        self.initPredictionConvolutions()
    
    def initPredictionConvolutions(self): #TODO: Initialize final layer bias to have essentially uniform logits on to avoid hockey stick loss
        """
        Initialize weights and biases with xavier uniform and zeros respectively
        """
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, conv4_3_features, conv7_features, conv8_2_features, conv9_2_features, conv10_2_features, conv11_2_features):
        """
        :param conv4_3_features: lower-level conv4_3 feature map, tensor of dimensions (N, 512, 38, 38)
        :param conv7_features: lower-level conv7 feature map, tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_features: higher-level conv8_2 feature map, tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_features: higher-level conv9_2 feature map, tensor of dimensions (N, 256, 5, 5)
        :param conv10_2_features: higher-level conv10_2 feature map, tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_features: higher-level conv11_2 feature map, tensor of dimensions (N, 256, 1, 1)
        :return: 8732 bounding box predictions and class scores for each image in batch
        """

        batchSize = conv4_3_features.size(0)

        #Predict localization bounding boxes (as offsets) for each of the 8732 prior boxes
        loc_conv4_3 = self.offset_conv4_3(conv4_3_features) #(N, 16, 38, 38), four offsets per prior, 16 predictions in total 
        loc_conv4_3 = loc_conv4_3.permute(0, 2, 3, 1).contiguous() #(N, 38, 38, 16), .view() requires contiguous chunk of memory
        loc_conv4_3 = loc_conv4_3.view(batchSize, -1, 4) #(N, 5776, 4), -1 ensures that total number of elements remains same. 
        #There are N images that each contain 5776 predicted localization bounding boxes (four priors every pixel of the 38x38 input), each box with 4 offsets

        loc_conv7 = self.offset_conv7(conv7_features) #(N, 24, 19, 19)
        loc_conv7 = loc_conv7.permute(0, 2, 3, 1).contiguous() #(N, 19, 19, 24)
        loc_conv7 = loc_conv7.view(batchSize, -1, 4) #(N, 2166, 4), 2166 boxes on this feature map

        loc_conv8_2 = self.offset_conv8_2(conv8_2_features) #(N, 24, 10, 10)
        loc_conv8_2 = loc_conv8_2.permute(0, 2, 3, 1).contiguous() #(N, 10, 10, 24)
        loc_conv8_2 = loc_conv8_2.view(batchSize, -1, 4) #(N, 600, 4), 690 prior boxes

        loc_conv9_2 = self.offset_conv9_2(conv9_2_features) #(N, 24, 5, 5)
        loc_conv9_2 = loc_conv9_2.permute(0, 2, 3, 1).contiguous() #(N, 5, 5, 24)
        loc_conv9_2 = loc_conv9_2.view(batchSize, -1, 4) #(N, 150, 4), 150 prior boxes

        loc_conv10_2 = self.offset_conv10_2(conv10_2_features) #(N, 16, 3, 3)
        loc_conv10_2 = loc_conv10_2.permute(0, 2, 3, 1).contiguous() #(N, 3, 3, 16)
        loc_conv10_2 = loc_conv10_2.view(batchSize, -1, 4) #(N, 36, 4), 36 prior boxes

        loc_conv11_2 = self.offset_conv11_2(conv11_2_features) #(N, 16, 1, 1)
        loc_conv11_2 = loc_conv11_2.permute(0, 2, 3, 1).contiguous() #(N, 1, 1, 16)
        loc_conv11_2 = loc_conv11_2.view(batchSize, -1, 4) #(N, 4, 4), 4 prior boxes

        #Predict classes in localization boxes for each of the 8732 prior boxes

        class_conv4_3 = self.class_conv4_3(conv4_3_features) #(N, 4 * numClasses, 38, 38). Each prior box contains numClasses class scores/probs
        class_conv4_3 = class_conv4_3.permute(0, 2, 3, 1).contiguous() #(N, 38, 38, 4 * numClasses)
        class_conv4_3 = class_conv4_3.view(batchSize, -1, self.numClasses) #(N, 5776, numClasses), 5776 boxes on this feature map

        class_conv7 = self.class_conv7(conv7_features) #(N, 6 * numClasses, 19, 19)
        class_conv7 = class_conv7.permute(0, 2, 3, 1).contiguous() #(N, 19, 19, 6 * numClasses)
        class_conv7 = class_conv7.view(batchSize, -1, self.numClasses) #(N, 2166, numClasses), 2166 boxes on this feature map

        class_conv8_2 = self.class_conv8_2(conv8_2_features) #(N, 6 * numClasses, 10, 10)
        class_conv8_2 = class_conv8_2.permute(0, 2, 3, 1).contiguous() #(N, 10, 10, 6 * numClasses)
        class_conv8_2 = class_conv8_2.view(batchSize, -1, self.numClasses) #(N, 600, numClasses), 600 boxes on this feature map

        class_conv9_2 = self.class_conv9_2(conv9_2_features) #(N, 6 * numClasses, 5, 5)
        class_conv9_2 = class_conv9_2.permute(0, 2, 3, 1).contiguous() #(N, 5, 5, 6 * numClasses)
        class_conv9_2 = class_conv9_2.view(batchSize, -1, self.numClasses) #(N, 150, numClasses), 150 boxes on this feature map

        class_conv10_2 = self.class_conv10_2(conv10_2_features) #(N, 4 * numClasses, 3, 3)
        class_conv10_2 = class_conv10_2.permute(0, 2, 3, 1).contiguous() #(N, 3, 3, 4 * numClasses)
        class_conv10_2 = class_conv10_2.view(batchSize, -1, self.numClasses) #(N, 36, numClasses), 36 boxes on this feature map

        class_conv11_2 = self.class_conv11_2(conv11_2_features) #(N, 4 * numClasses, 1, 1)
        class_conv11_2 = class_conv11_2.permute(0, 2, 3, 1).contiguous() #(N, 1, 1, 4 * numClasses)
        class_conv11_2 = class_conv11_2.view(batchSize, -1, self.numClasses) #(N, 4, numClasses), 4 boxes on this feature map

        #total of 8732 boxes, 8732 times numClasses class scores, 8732 times 4 offsets/predictions

        #Concatenate all predictions from different feature maps into one tensor, following order of prior boxes
        locs = torch.cat([loc_conv4_3, loc_conv7, loc_conv8_2, loc_conv9_2, loc_conv10_2, loc_conv11_2], dim=1) #(N, 8732, 4)
        classScores = torch.cat([class_conv4_3, class_conv7, class_conv8_2, class_conv9_2, class_conv10_2, class_conv11_2], dim=1) #(N, 8732, numClasses)

        return locs, classScores

class SSD300(nn.Module):
    """
    SSD300 in all its glory - VGG16 base model, auxiliary and prediction convolutions
    """

    def __init__(self, numClasses):
        super(SSD300, self).__init__()

        self.numClasses = numClasses

        self.base = VGG16Base()
        self.auxiliaryConvs = AuxiliaryConvolutions()
        self.predictionConvs = PredictionConvolutions(numClasses)

        #Lower level feature maps (conv4_3_features) have much larger scales. Take L2 normalization factor and rescale
        #Rescale factor initially set at 20, learned for each of conv4_3's 512 channels/filters during back-prop

        self.rescaleFactors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescaleFactors, 20)

        #Create prior boxes in center coordinate form
        self.priorsCenter = self.createPriorBoxes()

    def forward(self, images):
        """
        Forward prop. 

        :param images: images, tensor of dims (N, 3, 300, 300)
        :return: 8732 bounding box predictions (offsets) and class scores for each image
        """

        #Apply base VGG network layers (generate lower level feature maps)
        conv4_3_features, conv7_features = self.base(images) #(N, 512, 38, 38), (N, 1024, 19, 19)

        #Rescale conv4_3 after L2 norm
        norm = conv4_3_features.pow(2).sum(dim=1, keepdim=True).sqrt() #(N, 1, 38, 38)
        conv4_3_features = conv4_3_features / norm #(N, 512, 38, 38)
        conv4_3_features = conv4_3_features * self.rescaleFactors #(N, 512, 38, 38)

        #Apply auxiliary convolutions (generate higher level feature maps)
        conv8_2_features, conv9_2_features, conv10_2_features, conv11_2_features = \
        self.auxiliaryConvs(conv7_features) #(N, 512, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        #Run prediction convolutions (generate offsets for the 8732 prior boxes and class scores for each localization box encoded by the offsets)
        locs, classesScores = self.predictionConvs(conv4_3_features, conv7_features, conv8_2_features, 
                                                 conv9_2_features, conv10_2_features, conv11_2_features) #(N, 8732, 4), (N, 8732, numClasses)


        return locs, classesScores
    
    def createPriorBoxes(self):
        """
        Create the 8732 priors for the SSD300
        :return: prior boxes in center-size coordinates, tensor of dimensions (8732, 4)
        """

        featureMapDims = {'conv4_3': 38,
                        'conv7': 19,
                        'conv8_2': 10,
                        'conv9_2': 5,
                        'conv10_2': 3,
                        'conv11_2': 1}
        
        priorScales = {'conv4_3': 0.1,
                    'conv7': 0.2,
                    'conv8_2': 0.375,
                    'conv9_2': 0.55,
                    'conv10_2': 0.725,
                    'conv11_2': 0.9}
        
        aspectRatios = {'conv4_3': [1., 2., 0.5],
                        'conv7': [1., 2., 3., 0.5, .333],
                        'conv8_2': [1., 2., 3., 0.5, .333],
                        'conv9_2': [1., 2., 3., 0.5, .333],
                        'conv10_2': [1., 2., 0.5],
                        'conv11_2': [1., 2., 0.5]}
        
        featureMaps = list(featureMapDims.keys())

        priorBoxes = []

        for key, featureMap in enumerate(featureMaps):
            for i in range(featureMapDims[featureMap]):
                for j in range(featureMapDims[featureMap]):
                    #Compute center of prior box
                    cx = (j + 0.5)/featureMapDims[featureMap]
                    cy = (i + 0.5)/featureMapDims[featureMap]

                    #Iterate through aspect ratios, computing width and height for each ratio
                    #Width = scale * sqrt(aspect ratio)
                    #Height = scale / sqrt(aspect ratio)
                    for ratio in aspectRatios[featureMap]:
                        priorBoxes.append([cx, cy, priorScales[featureMap] * sqrt(ratio), priorScales[featureMap] / sqrt(ratio)])

                        #For aspect ratio of 1, add another prior box with scale equal to geometric mean of current feature and the next feature dims
                        if ratio == 1.:
                            try:
                                additionalScale = sqrt(priorScales[featureMap] * priorScales[featureMaps[key + 1]])
                            except IndexError:
                                additionalScale = 1.
                            priorBoxes.append([cx, cy, additionalScale, additionalScale])
            
        priorBoxes = torch.FloatTensor(priorBoxes) #Convert to tensor
        return priorBoxes

    def detectObjects(self, predictedLocs, predictedClassScores, minScore, maxOverlap, topK):
        """
        Decode the 8732 predicted localization offsets and class scores to detect objects

        For each class, perform non-maximum suppression (NMS) on candidate boxes above a min threshold 

        :param predictedLocs: predicted offsets/locations for each prior box, tensor of dimensions (N, 8732, 4)
        :param predictedClassScores: predicted class scores for each prior box, tensor of dimensions (N, 8732, numClasses)
        :param minScore: minimum threshold for a box to be considered a match for a certain class
        :param maxOverlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param topK: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch size
        """

        batchSize = predictedLocs.size(0)
        numPriorBoxes = self.priorsCenter.size(0)
        predictedClassScores = F.softmax(predictedClassScores, dim = 2) #(N, 8732, numClasses)

        allImagesBboxes = []
        allImagesLabels = []
        allImagesScores = []

        assert numPriorBoxes == predictedLocs.size(1) == predictedClassScores.size(1) #Should all equal 8732

        for i in range(batchSize):
            #Decode predicted localization offsets to get predicted bounding boxes in boundary box format
            decodedLocs = centerToBoundary(decodeOffsets(predictedLocs[i], self.priorsCenter)) #(8732, 4

            #Lists to store boxes and scores for this image
            imageBboxes = []
            imageLabels = []
            imageScores = []

            maxScores, bestLabel = predictedClassScores[i].max(dim=1) #(8732), (8732)

            for c in range(self.numClasses): #For each class:
                #Keep predicted boxes and scores where scores for a class are above min score
                classScores = predictedClassScores[i][:, c] #(8732)
                scoresAboveMinScore = classScores > minScore
                numAboveMinScore = scoresAboveMinScore.sum().item()
                if numAboveMinScore == 0:
                    continue
                qualifiedClassScores = classScores[scoresAboveMinScore] #(numQualified)
                qualifiedClassLocs = decodedLocs[scoresAboveMinScore] #(numQualified, 4)

                #Sort qualified predicted boxes and scores by scores
                qualifiedClassScores, sortIndices = qualifiedClassScores.sort(dim=0, descending=True) #(numQualified), (num > minScore)
                qualifiedClassLocs = qualifiedClassLocs[sortIndices] #(numQualified, 4)

                #Find overlap between predicted boxes
                overlap = findJaccardOverlap(qualifiedClassLocs, qualifiedClassLocs) #(numQualified, num > minScore)

                #Perform Non-Maximum Supression (NMS)
                #Tensor to keep track of boxes to suppress, 0 means don't suppress, 1 means suppress
                suppress = torch.zeros((numAboveMinScore), dtype=torch.uint8).to(device) #(num > minScore)

                for bbox in range(qualifiedClassLocs.size(0)):
                    #If current box is already marked for suppression, move on to next box
                    if suppress[bbox] == 1:
                        continue

                    #Suppress boxes whose overlaps (with current box) are greater than maxOverlap
                    suppress = torch.max(suppress, overlap[bbox] > maxOverlap) #(num > minScore)
                    #torch.max() retains previously suppressed boxes

                    #Don't suppress this box even though it has an overlap of 1 with itself
                    suppress[bbox] = 0
                #Store only unsuppressed boxes for this class (0s become 1s, 1s become 0s, this acts like a boolean mask)
                imageBboxes.append(qualifiedClassLocs[1 - suppress])
                imageLabels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                imageScores.append(qualifiedClassScores[1 - suppress])

            #If no object in any class is found, store a placeholder for 'background' class
            if len(imageBboxes) == 0:
                imageBboxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                imageLabels.append(torch.LongTensor([0]).to(device))
                imageScores.append(torch.FloatTensor([0.]).to(device))

            #Now that we've matched one box per object for this particular class, 
            #Concatenate into single tensors.
            imageBboxes = torch.cat(imageBboxes, dim=0) #(numObjects, 4)    
            imageLabels = torch.cat(imageLabels, dim=0) #(numObjects)
            imageScores = torch.cat(imageScores, dim=0) #(numObjects)
            numObjects = imageScores.size(0)

            #Keep only the top 'k' objects

            if numObjects > topK:
                imageScores, sortIndices = imageScores.sort(dim=0, descending=True)
                imageScores = imageScores[:topK] #(topK)
                imageBboxes = imageBboxes[sortIndices][:topK] #(topK, 4)
                imageLabels = imageLabels[sortIndices][:topK] #(topK)

            #Append a class's boxes, labels and scores to liss that contain all images' boxes, labels and scores
            allImagesBboxes.append(imageBboxes)
            allImagesLabels.append(imageLabels)
            allImagesScores.append(imageScores)

        return allImagesBboxes, allImagesLabels, allImagesScores #lists of length batch size

class MultiBoxLoss(nn.Module):
    """
    Multibox loss, common loss function for object detection. 

    Two parts:
    1. localization loss for predicted locations of boxes, 
    2. Confidence loss for predicted class scores 
    """
    def __init__(self, priorsCenter, threshold = 0.5, negPosRatio = 3, alpha = 1.):
        super(MultiBoxLoss, self).__init__()
        self.priorsCenter = priorsCenter
        self.priorsBoundary = centerToBoundary(priorsCenter)
        self.threshold = threshold
        self.negPosRatio = negPosRatio
        self.alpha = alpha

        self.smoothL1Loss = nn.SmoothL1Loss() 
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduce = False)

    def forward(self, predictedLocs, predictedClassScores, trueBboxes, trueLabels):
        """
        Forward prop.

        :param predictedLocs: predicted offsets/locations for each prior box, tensor of dimensions (N, 8732, 4) since there's 8732 prior boxes per image
        :param predictedClassScores: predicted class scores for each prior box, tensor of dimensions (N, 8732, numClasses)
        :param trueBboxes: true boundary boxes, list of N tensors
        :param trueLabels: true object labels, list of N tensors
        :return: multibox loss scalar
        """
        batchSize = predictedLocs.size(0)
        numPriorBoxes = self.priorsCenter.size(0)
        numClasses = predictedClassScores.size(2)
        
        assert numPriorBoxes == predictedLocs.size(1) == predictedClassScores.size(1)

        trueLocs = torch.zeros((batchSize, numPriorBoxes, 4), dtype=torch.float).to(device)
        trueClasses = torch.zeros((batchSize, numPriorBoxes), dtype=torch.long).to(device)

        for i in range(batchSize):
            numObjects = trueBboxes[i].size(0)

            overlaps = findJaccardOverlap(trueBboxes[i], self.priorsBoundary) #(numObjects, 8732) containing overlaps between all combinations of ground truths/prior boxes

            #Match each of the 8732 priors to object with which it has max overlap
            maxOverlapForEachPrior, objectForEachPrior = overlaps.max(dim=0) #(8732)

            #Ensure that all objects are represented in positive priors (non-background). In other words, we want to ensure that all ground truth objects are matched with a prior - 
            #1. An object might not be the best object for all priors so none of the priors would contain that object
            #2. All priors with the object could also be assigned as background priors based on the threshold of 0.5
            
            #Find prior with max overlap for each object
            _, maxPriorForEachObject = overlaps.max(dim=1) #(numObjects)
            #Assign each object to its max overlap prior
            objectForEachPrior[maxPriorForEachObject] = torch.LongTensor(range(numObjects)).to(device)

            #To ensure the max overlap priors qualify, artificially give these priors an overlap of greater than 0.5 for the object they're assigned to address 2.
            maxOverlapForEachPrior[maxPriorForEachObject] = 1.

            labelForEachPrior = trueLabels[i][objectForEachPrior] #(8732)
            #Priors whose overlaps with objects less than the threshold are set to be background (no object)
            labelForEachPrior[maxOverlapForEachPrior < self.threshold] = 0 #(8732)
            trueClasses[i] = labelForEachPrior

            trueLocs[i] = encodeOffsets(boundaryToCenter(trueBboxes[i][objectForEachPrior]), self.priorsCenter) #(8732, 4)

        #Identify non-background priors (positive priors)
        positivePriors = trueClasses != 0

        #Compute Localization loss only over positive (non-background priors)
        locLoss = self.smoothL1Loss(predictedLocs[positivePriors], trueLocs[positivePriors]) #scalar
        #Compute confidence loss over positive priors and most difficult (negative/hardest) negative priors for each image.
        #For each image, take the hardest negative priors, or priors with the greatest cross entropy loss, (negPosRatio * numPositives),
        #This is Hard Negative Mining - focuses on hardest negatives in each image, minimizes pos/neg imbalance
        
        numPositives = positivePriors.sum(dim=1) #(N)
        numHardNegatives = self.negPosRatio * numPositives #(N)
        #Find loss for all priors
        confLossAll = self.crossEntropyLoss(predictedClassScores.view(-1, numClasses), trueClasses.view(-1)) #(N * 8732)
        confLossAll = confLossAll.view(batchSize, numPriorBoxes) #(N, 8732)

        confLossPos = confLossAll[positivePriors] #(numPositives)
        
        #Find hard-negative priors (priors which the model performed the worst on)
        confLossNeg = confLossAll.clone() #(N, 8732)
        confLossNeg[positivePriors] = 0. #(N, 8732), set positive priors' loss to 0 since positive priors are never in the top n hard negatives
        confLossNeg, _ = confLossNeg.sort(dim = 1, descending = True) #(N, 8732), sort losses for each image in descending order/hardness
        hardnessRanks = torch.LongTensor(range(numPriorBoxes)).unsqueeze(0).expand_as(confLossNeg).to(device) #(N, 8732) 
        hardNegatives = hardnessRanks < numHardNegatives.unsqueeze(1) #(N, 8732), 1s for hard negatives, 0s for easy negatives
        confLossHardNeg = confLossAll[hardNegatives] #sum(hardNegatives), contains loss for hard negatives only

        #Average confidence loss over positive priors, compute both positive and hard-negative priors
        confLoss = (confLossHardNeg.sum() + confLossPos.sum()) / numPositives.sum().float() #scalar
    
        #Total loss - confidence loss and localization loss
        return confLoss + self.alpha*locLoss
    
def main():
    model = VGG16Base()
    print(model.state_dict().keys())

if __name__ == '__main__':
    main()