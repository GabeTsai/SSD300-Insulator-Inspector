import numpy as np
import random
import os
import json
import sys
from DataTransforms import createBackgroundImages


labelMap = {'Background':0, 'Insulator':1, 'Flashover damage':2, 'Broken':3, 'No issues':4, 'notbroken-notflashed':4}

with open ('/Users/HP/Documents/InsulatorInspector/Data/Train/labels.json', 'r') as f:
    labels = json.load(f)

trainImageNames = os.listdir('/Users/HP/Documents/InsulatorInspector/Data/Train/Images')
trainImages = []
trainObjectDict = {'bbox':[], 'labels':[]} #bbox is a list of float tensors, #category is a list of long tensors
trainObjects = [trainObjectDict.copy() for i in range(len(trainImageNames)-1)]


for i in range(len(trainImageNames) - 1):
    trainImages.append(os.path.join('/Users/HP/Documents/InsulatorInspector/Data/Train/Images', labels[i]['filename']))
    objectList = labels[i]['Labels']['objects'] #list of objects for an image

    bboxes_list = []
    labels_list = []

    for object in objectList: #each object is a dictionary
        bbox = object['bbox']
        bboxes_list.append(bbox)

        #Append label
        if (object['string'] == 1):
            labels_list.append(int(labelMap['Insulator']))
        else:
            condition = object['conditions']
            if (any('notbrokennotflashed' in key.replace('-', '').lower() for key in condition.keys()) or 'No issues' in condition.keys()):
                labels_list.append(int(labelMap['notbroken-notflashed']))
            elif ('glaze' in condition.keys()):
                labels_list.append(int(labelMap[condition['glaze']]))
            else:
                labels_list.append(int(labelMap[condition['shell']]))

    trainObjects[i]['bbox'] = bboxes_list
    trainObjects[i]['labels'] = labels_list

assert len(trainImages) == len(trainObjects)
print(len(trainImages), len(trainObjects))
#Generate Background Images
# print(len(trainObjects), len(trainImages))
# createBackgroundImages(trainImages, '/Users/HP/Documents/InsulatorInspector/Data/Train/Images', 0.1, trainObjects)
# print(len(trainObjects), len(trainImages))


#Do Train/Test split
dataLists = list(zip(trainImages, trainObjects))
print(len(dataLists))
testSize = int(len(trainImages)*0.1)
valSize = int(len(trainImages)*0.1)
random.shuffle(dataLists)

testLists = dataLists[:testSize]
valLists = dataLists[testSize:testSize+valSize]
trainLists = dataLists[testSize+valSize:]

print(len(testLists), len(valLists), len(trainLists))
testImages, testObjects = zip(*testLists)
trainImages, trainObjects = zip(*trainLists)    
valImages, valObjects = zip(*valLists)

updatedTestImages = []
for image in testImages:    
    newImagePath = image.replace('Train/Images', 'Test')
    os.replace(image, newImagePath)
    updatedTestImages.append(newImagePath)

updatedValImages = []
for image in valImages:
    newImagePath = image.replace('Train/Images', 'Val')
    os.replace(image, newImagePath)
    updatedValImages.append(newImagePath)

print(len(updatedValImages), len(updatedTestImages))
assert len(updatedValImages) == len(updatedTestImages) 

with open(os.path.join('/Users/HP/Documents/InsulatorInspector/Data/ProcessedData', 'Train_Images.json'), 'w') as f:
    json.dump(trainImages, f)

with open(os.path.join('/Users/HP/Documents/InsulatorInspector/Data/ProcessedData', 'Train_Objects.json'), 'w') as f:
    json.dump(trainObjects, f)

with open(os.path.join('/Users/HP/Documents/InsulatorInspector/Data/ProcessedData', 'Val_Images.json'), 'w') as f:
    json.dump(updatedValImages, f)

with open(os.path.join('/Users/HP/Documents/InsulatorInspector/Data/ProcessedData', 'Val_Objects.json'), 'w') as f:
    json.dump(valObjects, f)

with open(os.path.join('/Users/HP/Documents/InsulatorInspector/Data/ProcessedData', 'Test_Images.json'), 'w') as f:
    json.dump(updatedTestImages, f)

with open(os.path.join('/Users/HP/Documents/InsulatorInspector/Data/ProcessedData', 'Test_Objects.json'), 'w') as f:
    json.dump(testObjects, f)
