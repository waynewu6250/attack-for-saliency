import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from tqdm import tqdm

from data import ImageSet
from config import opt

def train_models(img_path, img_feature_path):
    
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    #Data
    data = ImageSet(opt.data_path, opt.label_path)
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=False)

    #Model
    model = tv.models.resnet50(pretrained=True)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model.to(device)

    optimizer = Adam(model.parameters, lr=opt.lr)
    criterion = t.nn.CrossEntropyLoss()

    for epoch in opt.epochs:
        for ii, (imgs, indexes, labels) in tqdm.tqdm(enumerate(dataloader)):

            imgs = Variable(imgs).to(device)
            labels = Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (epoch+1) % opt.save_model == 0:
                print("Training loss:" ,loss.item())
                t.save(model.state_dict(), "checkpoints/model-{}.pth".format(epoch))