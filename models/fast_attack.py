import os
import numpy as np

import torch as t
from torch import nn
from torch.autograd import Variable
import torchvision as tv
from PIL import Image

class UntargetedAttack():

    def __init__(self, model, alpha, image_path, true_label):
        
        self.model = model
        self.model.eval()
        self.alpha = alpha

        # image
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert('RGB')
        self.img = self.transform(img)
        self.img.grad = 0
        self.true_label = Variable(true_label)
    
    def generate(self):

        criterion = t.nn.CrossEntropyLoss()
        output = self.model(self.img)

        pred_loss = criterion(output, self.true_label)
        




