import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from DataTransforms import transform

class InsulatorDataset(Dataset):
    def __init__(self, dataFolder, split):
        """
        :param dataFolder: folder where json data files stored
        :param split: 'train' or 'test'
        """
        self.split = split.lower()

        assert self.split in {'train', 'test', 'val'}

        self._dataFolder = dataFolder

        with open(os.path.join(dataFolder, self.split + '_Images.json'), 'r') as f:
            self._images = json.load(f)
        
        with open(os.path.join(dataFolder, self.split + '_Objects.json'), 'r') as f:
            self._objects = json.load(f)
        
        assert len(self._images) == len(self._objects)
    
    def __getitem__(self, i):
        """
        Fetch images, bboxes and labels for a batch.
        """
        #Read image
        image = Image.open(self._images[i], mode = 'r')
        image = image.convert('RGB')

        #Read objects for image
        objects = self._objects[i]
        bboxes = torch.FloatTensor(objects['bbox'])
        labels = torch.LongTensor(objects['labels'])

        #Apply transformations
        image, bboxes, labels = transform(image, bboxes, labels, self.split)

        return image, bboxes, labels
    
    def __len__(self):
        return len(self._images)
    
    def collateFunc(self, batch):
        """
        Images contain a different number of objects, so they must be collated using lists.

        :param batch: iterable of N images,bboxes,labels from __getitem__()
        """
        images = list()
        bboxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            bboxes.append(b[1])
            labels.append(b[2])
        
        images = torch.stack(images, dim = 0)

        return images, bboxes, labels 
    
