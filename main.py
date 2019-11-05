import random
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
import torchvision as tv
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

from data import ImageSet
from config import opt

def load_pickle(file):
    with open(file,'rb') as f:
        return pickle.load(f)

def train():
    
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
            outputs = model(imgs)

            val_corrects += t.sum(t.max(outputs, 1)[1] == labels)
            #val_loss = criterion(outputs, labels)
        
        #print("Validation loss: {:.4f}".format(val_loss.item()))
        print("Validation accuracy: {:.4f}".format(val_corrects.double() / val_data.num_data))
        print()

        if (epoch+1) % opt.save_model == 0:
            t.save(model.state_dict(), "checkpoints/model-{}.pth".format(epoch))
            

if __name__ == "__main__":
    train()