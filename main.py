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

def train(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    #Data
    imgs = load_pickle(opt.data_path)
    labels = load_pickle(opt.label_path)
    labels = [labels[img] for img in imgs]
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.1, random_state = 42)

    train_data = ImageSet(X_train, y_train, opt.dic_path, load=True)
    val_data = ImageSet(X_test, y_test, opt.dic_path, load=True)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True)

    #Model
    model = tv.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_data.num_labels)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model.to(device)
    

    optimizer = Adam(model.parameters(), lr=opt.lr)
    criterion = t.nn.CrossEntropyLoss()

    
    for epoch in range(opt.epochs):
        print("=========== epoch {} ===========".format(epoch))
        train_corrects = 0
        # Training Phase
        for ii, (imgs, indexes, labels) in tqdm(enumerate(train_dataloader)):

            imgs = Variable(imgs).to(device)
            labels = Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_corrects += t.sum(t.max(outputs, 1)[1] == labels)
            print(loss.item())
        
        print("Training loss: {:.4f}".format(loss.item()))
        print("Training accuracy: {:.4f}".format(train_corrects.double() / train_data.num_data))
        
        # Validation Phase
        val_corrects = 0
        for ii, (imgs, indexes, labels) in enumerate(val_dataloader):

            imgs = Variable(imgs).to(device)
            labels = Variable(labels).to(device)
            with t.no_grad():
                outputs = model(imgs)

            val_corrects += t.sum(t.max(outputs, 1)[1] == labels)
            val_loss = criterion(outputs, labels)
        
        print("Validation loss: {:.4f}".format(val_loss.item()))
        print("Validation accuracy: {:.4f}".format(val_corrects.double() / val_data.num_data))
        print()

        if (epoch+1) % opt.save_model == 0:
            t.save(model.state_dict(), "checkpoints/model-{}.pth".format(epoch))

    #############################################

def test(i):

    # for k, v in kwargs.items():
    #     setattr(opt, k, v)

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    #Data
    imgs = load_pickle(opt.data_path)
    labels = load_pickle(opt.label_path)
    label2id = load_pickle(opt.dic_path)
    id2label = {v:k for k, v in label2id.items()}

    labels = [labels[img] for img in imgs]
    index = np.random.randint(0,len(imgs), 1)[0]
    transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])

    img = Image.open(imgs[index]).convert('RGB')
    img.save("/nethome/twu367/attack-for-saliency/generated/original/original_classified_img_{}.jpg".format(i))
    img = transform(img)

    #Model
    model = tv.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(label2id))
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
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
    P.imsave("/nethome/twu367/attack-for-saliency/generated/original/gradient_{}.png".format(i), image_2d, cmap=P.cm.gray, vmin=0, vmax=1)

    # smooth grad
    print(img.shape)
    dydx = get_smooth_mask(img)
    image_2d = convert_gray(dydx.unsqueeze(0))
    P.imsave("/nethome/twu367/attack-for-saliency/generated/original/smooth_gradient_{}.png".format(i), image_2d, cmap=P.cm.gray, vmin=0, vmax=1)
    

    
    print()
    print("predicted label: ", id2label[t.max(outputs,1)[1].item()])
    print("Real label: ", labels[index])

    ########## for testing ##########
    # print("Testing Phase")
    # img_before = Image.open("untargeted_original.jpg").convert('RGB')
    # img_after = Image.open("untargeted_adv_img.jpg").convert('RGB')
    # img_before = transform(img_before)
    # img_after = transform(img_after)

    # img_before = Variable(img_before).to(device)
    # img_after= Variable(img_after).to(device)
    # outputs_before = model(img_before.unsqueeze(0))
    # outputs_after = model(img_after.unsqueeze(0))
    # print("Label before attack: ", t.max(outputs_before,1)[1].item())
    # print("Label after attack: ", t.max(outputs_after,1)[1].item())



    
    #############################################

def perform_attack(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    #Data
    imgs = load_pickle(opt.data_path)
    labels = load_pickle(opt.label_path)
    label2id = load_pickle(opt.dic_path)
    id2label = {v:k for k, v in label2id.items()}

    labels = [labels[img] for img in imgs]
    index = 3510 #np.random.randint(0,len(imgs), 1)[0] #3510
    

    #Model
    model = tv.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(label2id))
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model.eval()
    model.to(device)

    # Select attack model
    path = '/nethome/twu367/attack-for-saliency/'+imgs[index]
    if opt.attack_model == "untargeted-attack":
        attackmodel = UntargetedAttack(model, opt.alpha, path, label2id[labels[index]], device)
    elif opt.attack_model == "targeted-attack":
        target_label = 10
        attackmodel = TargetedAttack(model, opt.alpha, path, label2id[labels[index]], device, target_label)
    elif opt.attack_model == "mask-targeted-attack":
        target_label = 10
        mask_model = SimpleUnet()
        attackmodel = MaskTargetedAttack(model, mask_model, opt.alpha, path, label2id[labels[index]], device, target_label)
    
    netg = Generator(opt.inf, opt.gnf)
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location='cpu'))
    netg.to(device)
    
    return attackmodel.generate(netg, opt)

            

if __name__ == "__main__":
    # import fire
    # fire.Fire()

    # for i in range(3):
    #     test(i)
    perform_attack()
    # iteration = sum([perform_attack() for i in range(10)]) / 10
    # print(iteration)