import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
import torchvision as tv
import torch.nn.functional as F

from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from data import ImageSet
from config import opt
from models import UntargetedAttack, TargetedAttack, MaskTargetedAttack
from models import Generator, SimpleUnet

def load_pickle(file):
    with open(file,'rb') as f:
        return pickle.load(f)

def test(i):
    
    # for k, v in kwargs.items():
    #     setattr(opt, k, v)

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    #Data
    with open("imagenet/imagenet1000_clsidx_to_labels.txt") as f:
        idx2label = eval(f.read())
    
    transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    img = Image.open('imagenet/elephants.jpg').convert('RGB')
    img = transform(img)

    #Model
    model = tv.models.resnet50(pretrained=True)
    model.eval()
    model.to(device)
    
    # Compute Grads: saliency map
    def get_mask(img):
        
        img = Variable(img).requires_grad_(True).to(device)
        outputs = model(img.unsqueeze(0))
        y = outputs[0][t.max(outputs,1)[1].item()]
        dydx = t.autograd.grad(outputs=y,
                            inputs=img,
                            grad_outputs=t.ones(y.size()).to(device),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
        return dydx, outputs
    
    def convert_gray(dydx):
        image_2d = np.sum(np.abs(dydx[0].cpu().detach().numpy()), axis=0)
        vmax = np.percentile(image_2d, 99)
        vmin = np.min(image_2d)
        return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    
    def get_smooth_mask(img, stdev_spread = .15, nsamples = 25, magnitude = True):
        
        stdev = stdev_spread * (t.max(img) - t.min(img))

        total_gradients = t.zeros(img.shape).to(device)
        for _ in range(nsamples):
            noise = t.FloatTensor(np.random.normal(0, stdev, img.shape))
            x_plus_noise = img + noise
            grad, _ = get_mask(x_plus_noise)
            
            if magnitude:
                total_gradients += (grad[0] * grad[0])
            else:
                total_gradients += grad[0]

        return total_gradients / nsamples

    
    # Rough grad
    dydx, outputs = get_mask(img)
    image_2d = convert_gray(dydx)
    from matplotlib import pylab as P
    P.imsave("/nethome/twu367/attack-for-saliency/imagenet/gradient.png", image_2d, cmap=P.cm.gray, vmin=0, vmax=1)

    # smooth grad
    dydx = get_smooth_mask(img)
    image_2d = convert_gray(dydx.unsqueeze(0))
    P.imsave("/nethome/twu367/attack-for-saliency/imagenet/smooth_gradient.png", image_2d, cmap=P.cm.gray, vmin=0, vmax=1)
    
    print("predicted label: ", idx2label[t.max(outputs,1)[1].item()])

def perform_attack(**kwargs):
    
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    #Data
    with open("imagenet/imagenet1000_clsidx_to_labels.txt") as f:
        idx2label = eval(f.read())
    
    transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    img = Image.open('imagenet/elephants.jpg').convert('RGB')
    img = transform(img)

    #Model
    model = tv.models.resnet50(pretrained=True)
    model.eval()
    model.to(device)

    img = Variable(img).requires_grad_(True).to(device)
    outputs = model(img.unsqueeze(0))
    y = t.max(outputs,1)[1].item()
    print(idx2label[101])

    # Select attack model
    path = 'imagenet/elephants.jpg'
    if opt.attack_model == "untargeted-attack":
        attackmodel = UntargetedAttack(model, opt.alpha, path, y, device)
    elif opt.attack_model == "targeted-attack":
        target_label = 101
        attackmodel = TargetedAttack(model, opt.alpha, path, y, device, target_label)
    elif opt.attack_model == "mask-targeted-attack":
        target_label = 101
        mask_model = SimpleUnet()
        attackmodel = MaskTargetedAttack(model, mask_model, opt.alpha, path, y, device, target_label)
    
    netg = Generator(opt.inf, opt.gnf)
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location='cpu'))
    netg.to(device)
    
    return attackmodel.generate(netg, opt)

if __name__ == "__main__":
    #test(0)
    perform_attack()