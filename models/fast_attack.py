import os
import numpy as np

import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
import torchvision as tv
from torchvision.utils import save_image
from PIL import Image
import cv2
import copy

class UntargetedAttack():

    def __init__(self, model, alpha, image_path, true_label, device):
        
        self.model = model
        self.alpha = alpha
        self.device = device

        # image
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert('RGB')
        img.save('original.jpg')
        self.img = self.transform(img)
        self.img.unsqueeze_(0)

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
        recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0])
        
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
        
        #recreated_im[recreated_im > 1] = 1
        #recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)

        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        
        # Convert RBG to GBR
        #recreated_im = recreated_im[..., ::-1]
        return recreated_im

    
    def generate(self, netg, opt):
        
        adv_noise = t.randn(1, opt.inf, 1, 1).to(self.device)
        optimizer_g = Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
        criterion = t.nn.CrossEntropyLoss().to(self.device)

        img_original = Variable(self.img).to(self.device)
        img_as_var = Variable(self.img).to(self.device)

        for i in range(100):

            print("\n======== Iteration {} ========".format(i))
            print('Original image was classified as: ', self.true_label_var.item())

            optimizer_g.zero_grad()

            # First pass the original image into model
            output = self.model(img_original)
            
            pred_loss = criterion(output, self.true_label_var)
            print("Original loss: ", pred_loss.item())
            prediction_true = t.max(output,1)[1].item()
            
            # Create noise
            # adv_noise.copy_(t.randn(1, opt.inf, 1, 1))
            fake_img = netg(adv_noise)
            
            #adv_noise = self.alpha * t.sign(img.grad.data)
            img_with_noise = img_as_var + self.alpha*fake_img
            #img_reconstruct = self.recreate_image(img_with_noise)
            #self.img_as_var = self.transform(t.Tensor(img_reconstruct).to(self.device))

            # Re pass the processes image into model
            output_reconstruct = self.model(img_with_noise)
            pred_loss_reconstruct = -criterion(output_reconstruct, self.true_label_var)
            print("Later loss: ", pred_loss_reconstruct.item())

            pred_loss_reconstruct.backward(retain_graph=True)
            optimizer_g.step()

            prediction = t.max(output_reconstruct,1)[1].item()
            confirmation_score = F.softmax(output_reconstruct[0], dim=0)[prediction]
            

            if prediction != self.true_label:
                print('\nAttack Success!!')
                print('Original image was predicted as: ', prediction_true)
                print('With adversarial noise converted to: ', prediction)
                print('The confident score by probability is: ', confirmation_score.item())

                im = Image.fromarray(self.recreate_image(img_original))
                im.save('untargeted_original.jpg')
                im = Image.fromarray(self.recreate_image(fake_img))
                im.save('untargeted_adv_noise.jpg')
                im = Image.fromarray(self.recreate_image(img_with_noise))
                im.save('untargeted_adv_img.jpg')

                # save_image(img_original, 'untargeted_original.jpg')
                # save_image(self.alpha*fake_img, 'untargeted_adv_noise.jpg')
                # save_image(img_as_var, 'untargeted_adv_img.jpg')

                if confirmation_score.item() > 0.9:
                    return i
                    break

        return 1










