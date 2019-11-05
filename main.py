import random
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
import torchvision as tv
from tqdm import tqdm

from data import ImageSet
from config import opt

def train():
    
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    #Data
    train_data = ImageSet(opt.data_path, opt.label_path, opt.dic_path, load=True)
    val_data = ImageSet(opt.val_path, opt.val_label_path, opt.dic_path, load=True)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True)

    #Model
    model = tv.models.resnet50(pretrained=True)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=opt.lr)
    criterion = t.nn.CrossEntropyLoss()

    
    for epoch in range(opt.epochs):
        
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
        
        print("Training loss: {:.4f}".format(loss.item()))
        print("Training accuracy: {:.4f}".format(train_corrects.double() / train_data.num_data))
        
        # Validation Phase
        val_corrects = 0
        for ii, (imgs, indexes, labels) in enumerate(val_dataloader):

            imgs = Variable(imgs).to(device)
            labels = Variable(labels).to(device)
            outputs = model(imgs)

            val_corrects += t.sum(t.max(outputs, 1)[1] == labels)
            val_loss = criterion(outputs, labels)
        
        print("Validation loss: {:.4f}".format(val_loss.item()))
        print()

        if (epoch+1) % opt.save_model == 0:
            t.save(model.state_dict(), "checkpoints/model-{}.pth".format(epoch))
            

if __name__ == "__main__":
    train()