import os
import numpy as np

import torch as t
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
import cv2

class UntargetedAttack():

    def __init__(self, model, alpha, image_path, true_label, device):
        
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
        self.img = Variable(self.img).to(device)

        self.img_original = img
        self.true_label = true_label
        self.true_label_var = Variable(t.LongTensor([true_label])).to(device)

    
    def recreate_image(self, im_as_var):
        """
            Recreates images from a torch variable, sort of reverse preprocessing
        Args:
            im_as_var (torch variable): Image to recreate
        returns:
            recreated_im (numpy arr): Recreated image in array
        """

        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1/0.229, 1/0.224, 1/0.225]
        recreated_im = copy.copy(im_as_var.data.numpy()[0])
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)

        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        
        # Convert RBG to GBR
        recreated_im = recreated_im[..., ::-1]
        return recreated_im
    
    def create_noise(self, img):

        adv_noise = self.alpha * t.sign(img.grad.data)
        img.data = img.data + adv_noise

        return img

    
    def generate(self):
        
        criterion = t.nn.CrossEntropyLoss()

        for _ in range(10):

            self.img.grad = None

            # First pass the original image into model
            output = self.model(self.img.unsqueeze(0))
            pred_loss = criterion(output, self.true_label_var)
            pred_loss.backward()

            # Create noise
            img_with_noise = self.create_noise(self.img)
            img_reconstruct = self.recreate_image(img_with_noise)
            self.img = self.transform(img_reconstruct)

            # Re pass the processes image into model
            output = self.model(self.img.unsqueeze(0))
            prediction = t.max(outputs,1)[1].item()
            confirmation_score = F.softmax(output)[prediction]

            if prediction != self.true_label:
                print('Original image was predicted as: ', self.true_label)
                print('With adversarial noise converted to: ', prediction)
                print('The confident score by probability is: ', confirmation_score)

                noise_image = self.img_original - self.img
                cv2.imwrite('../generated/untargeted_adv_noise_from_' + str(self.true_label) + '_to_' +
                            str(prediction) + '.jpg', noise_image)
                
                cv2.imwrite('../generated/untargeted_adv_img_from_' + str(self.true_label) + '_to_' +
                            str(prediction) + '.jpg', self.img)
        return 1










