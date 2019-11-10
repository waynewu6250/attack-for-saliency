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
from models import UntargetedAttack
from models import Generator

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

def test(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)

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

    # img = Image.open(imgs[index]).convert('RGB')
    img = Image.open("untargeted_adv_img.jpg").convert('RGB')
    img = transform(img)

    #Model
    model = tv.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(label2id))
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model.eval()
    model.to(device)
    
    img = Variable(img).to(device)
    outputs = model(img.unsqueeze(0))
    
    # print("predicted label: ", id2label[t.max(outputs,1)[1].item()])
    print("predicted label: ", t.max(outputs,1)[1].item())
    print("Real label: ", labels[index])

    
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
    index = np.random.randint(0,len(imgs), 1)[0]

    #Model
    model = tv.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(label2id))
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model.eval()
    model.to(device)

    # Select attack model
    if opt.attack_model == "fast_attack":
        attackmodel = UntargetedAttack(model, opt.alpha, '/nethome/twu367/attack-for-saliency/'+imgs[index], label2id[labels[index]], device)
    netg = Generator(opt.inf, opt.gnf)
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location='cpu'))
    netg.to(device)
    
    attackmodel.generate(netg, opt)

            

if __name__ == "__main__":
    # import fire
    # fire.Fire()
    perform_attack()
    test()