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

class MaskTargetedAttack():

    def __init__(self, model, mask_model, alpha, image_path, true_label, device, target_label):
        
        self.model = model
        self.mask_model = mask_model.to(device)
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

        self.target_label = target_label
        self.target_label_var = Variable(t.LongTensor([target_label])).to(device)
    
    ##########################################################
    ##########################################################
    # Compute Grads: saliency map
    def get_mask(self, img):
        
        img = Variable(img).requires_grad_(True).to(self.device)
        outputs = self.model(img)
        y = outputs[0][t.max(outputs,1)[1].item()]
        dydx = t.autograd.grad(outputs=y,
                            inputs=img,
                            grad_outputs=t.ones(y.size()).to(self.device),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
        return dydx, outputs
    
    def convert_gray(self, dydx):
        image_2d = np.sum(np.abs(dydx[0].cpu().detach().numpy()), axis=0)
        vmax = np.percentile(image_2d, 99)
        vmin = np.min(image_2d)
        return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    
    def get_smooth_mask(self, img, stdev_spread = .15, nsamples = 25, magnitude = True):
        
        stdev = stdev_spread * (t.max(img) - t.min(img))

        total_gradients = t.zeros(img.shape).to(self.device)
        for _ in range(nsamples):
            noise = t.FloatTensor(np.random.normal(0, stdev, img.shape))
            x_plus_noise = img + noise
            grad, _ = self.get_mask(x_plus_noise)
            
            if magnitude:
                total_gradients += (grad[0] * grad[0])
            else:
                total_gradients += grad[0]

        return total_gradients / nsamples
    ##########################################################
    ##########################################################

    
    def recreate_image(self, im_as_var, noise=False):
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
            if not noise:
                recreated_im[c] -= reverse_mean[c]
        
        #recreated_im[recreated_im > 1] = 1
        #recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)
        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        
        # Convert RBG to GBR
        # recreated_im = recreated_im[..., ::-1]
        return recreated_im
    
    def create_mask(self, img_original, shape, start_x, start_y, width):
        mask = t.zeros(shape).to(self.device)
        x = range(start_x,start_x+width)
        y = range(start_y,start_y+width)
        xv, yv = np.meshgrid(x, y)
        for i in range(3):
            mask[0][i][xv, yv] = 1
        return mask
    
    def run_iterations(self, netg, adv_noise, optimizer_g, criterion, img_original, img_as_var, mask_in, mask_target=None, unet=False):
        
        im_original, im_noise, im_adv = None, None, None
        optimizer_q = Adam(netg.parameters())

        for i in range(500):
    
            print("\n======== Iteration {} ========".format(i))
            print('Original image was classified as: ', self.true_label_var.item())

            optimizer_g.zero_grad()
            optimizer_q.zero_grad()

            # First pass the original image into model
            output = self.model(img_original)
            
            pred_loss = criterion(output, self.true_label_var)
            print("Original loss: ", pred_loss.item())
            prediction_true = t.max(output,1)[1].item()
            
            fake_img = netg(adv_noise)
            
            if unet:
                # mask = self.mask_model(mask_in)
                # mask = F.interpolate(mask, size=[224, 224], mode="bilinear")
                mask_loss = nn.MSELoss()
                
                upsample = t.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
                upsampled_mask = upsample(mask_in)
                mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3)).to(self.device)

            else:
                mask = mask_in
            print(mask)
            #adv_noise = self.alpha * t.sign(img.grad.data)
            img_with_noise = img_as_var + self.alpha*fake_img*mask
            #img_reconstruct = self.recreate_image(img_with_noise)
            #self.img_as_var = self.transform(t.Tensor(img_reconstruct).to(self.device))

            # Re pass the processes image into model
            output_reconstruct = self.model(img_with_noise)
            prediction = t.max(output_reconstruct,1)[1].item()
            confirmation_score = F.softmax(output_reconstruct[0], dim=0)[prediction]

            if unet:
                def tv_norm(input, tv_beta):
                    img = input[0, 0, :]
                    row_grad = t.mean(t.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
                    col_grad = t.mean(t.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
                    return row_grad + col_grad
                pred_loss_reconstruct = confirmation_score + 0.1*t.mean(t.abs(1-mask)) + 0.2*tv_norm(mask, 3) + 0.1*mask_loss(mask, mask_target)
            else:
                pred_loss_reconstruct = criterion(output_reconstruct, self.target_label_var)
            print("Later loss: ", pred_loss_reconstruct.item())

            pred_loss_reconstruct.backward(retain_graph=True)
            optimizer_g.step()
            optimizer_q.step()

            if prediction != self.true_label:
                print('\nAttack Success!!')
                print('Original image was predicted as: ', prediction_true)
                print('With adversarial noise converted to: ', prediction)
                print('The confident score by probability is: ', confirmation_score.item())

                im_original = self.recreate_image(img_original)
                im_noise = self.recreate_image(fake_img*mask, noise=True)
                im_adv = self.recreate_image(img_with_noise)

                if pred_loss_reconstruct.item() < 0.5:
                    break

            # if prediction == self.target_label:
            #     print('\nAttack Success!!')
            #     print('Original image was predicted as: ', prediction_true)
            #     print('With adversarial noise converted to: ', prediction)
            #     print('The confident score by probability is: ', confirmation_score.item())

            #     im_original = self.recreate_image(img_original)
            #     im_noise = self.recreate_image(self.alpha*fake_img*mask, noise=True)
            #     im_adv = self.recreate_image(img_with_noise)

            #     if confirmation_score.item() > 0.9:
            #         break
        
        return i, confirmation_score, im_original, im_noise, im_adv


    def generate(self, netg, opt):
        
        adv_noise = t.randn(1, opt.inf, 1, 1).to(self.device)
        criterion = t.nn.CrossEntropyLoss().to(self.device)
        optimizer_g = Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
        img_original = Variable(self.img).to(self.device)
        img_as_var = Variable(self.img).to(self.device)

        if opt.mode == "partial":

            answer = []
            imgs = []
            width = 200
            step = 50
            for start_x in range(0, 224, step):
                if start_x + width >= 224:
                    start_x = 224-width
                for start_y in range(0, 224, step):
                    if start_y + width >= 224:
                        start_y = 224-width
                    mask = self.create_mask(img_original, img_original.shape, start_x, start_y, width)
                    stop_index, confirmation_score, im_original, im_noise, im_adv = self.run_iterations(netg, adv_noise, optimizer_g, criterion, img_original, img_as_var, mask)
                    if stop_index != 0 and stop_index != 1:
                        imgs.append((im_original, im_noise, im_adv))
                        answer.append((start_x, start_y, stop_index, confirmation_score.item()))
            
            for start_x, start_y, stop_index, score in answer:
                print("The position ({}, {}) has stop index at {} with score {:.03f}".format(start_x, start_y, stop_index, score))
            best_noise_index = np.argmin([content[2] for content in answer])
            im = Image.fromarray(imgs[best_noise_index][0])
            im.save('mask_targeted_original.jpg')
            im = Image.fromarray(imgs[best_noise_index][1])
            im.save('mask_targeted_adv_noise.jpg')
            im = Image.fromarray(imgs[best_noise_index][2])
            im.save('mask_targeted_adv_img.jpg')
        
        elif opt.mode == "random":
            
            # shape = img_original.shape
            # mask = t.LongTensor(np.random.binomial(n=1,p=0.1,size=(shape[-1], shape[-1]))).to(self.device)
            # mask = mask.repeat(1,3,1,1)
            
            dydx = self.get_smooth_mask(self.img)
            mask = self.convert_gray(dydx)
            mask[mask<=0.1]=0
            from matplotlib import pylab as P
            P.imsave("mask.jpg", mask, cmap=P.cm.gray, vmin=0, vmax=1)
            
            #mask = np.random.permutation(mask)
            #mask = np.random.permutation(mask.T)
            mask = t.Tensor(mask).to(self.device)
            
            #mask = mask.permute(1,0)
            
            stop_index, confirmation_score, im_original, im_noise, im_adv = self.run_iterations(netg, adv_noise, optimizer_g, criterion, img_original, img_as_var, mask)
            

            im = Image.fromarray(im_original)
            im.save('mask_targeted_original.jpg')
            im = Image.fromarray(im_noise)
            im.save('mask_targeted_adv_noise.jpg')
            im = Image.fromarray(im_adv)
            im.save('mask_targeted_adv_img.jpg')
        
        elif opt.mode == "unet":
            dydx = self.get_smooth_mask(self.img)
            mask = self.convert_gray(dydx)
            mask[mask<=0.1]=0
            from matplotlib import pylab as P
            P.imsave("mask.jpg", mask, cmap=P.cm.gray, vmin=0, vmax=1)
            
            mask_target = t.Tensor(mask).to(self.device)
            # upsample = t.nn.UpsamplingBilinear2d(size=(572, 572)).cuda()
            # mask_in = upsample(img_original)
            #params = list(self.mask_model.parameters())+list(netg.parameters())
            #optimizer_g = Adam(self.mask_model.parameters(), 0.01, betas=(opt.beta1, 0.999), weight_decay=2)

            # Create noise
            adv_noise.copy_(t.randn(1, opt.inf, 1, 1))
            mask_in = Variable(t.ones(1,1,28,28), requires_grad=True)
            optimizer_g = Adam([mask_in], 0.01, betas=(opt.beta1, 0.999))


            
            stop_index, confirmation_score, im_original, im_noise, im_adv = self.run_iterations(netg, adv_noise, optimizer_g, criterion, img_original, img_as_var, mask_in, mask_target, unet=True)

            im = Image.fromarray(im_original)
            im.save('mask_targeted_original.jpg')
            im = Image.fromarray(im_noise)
            im.save('mask_targeted_adv_noise.jpg')
            im = Image.fromarray(im_adv)
            im.save('mask_targeted_adv_img.jpg')




        return 1