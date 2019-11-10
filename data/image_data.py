import torch as t
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import tqdm
import pickle

class ImageSet(Dataset):

    def __init__(self, imgs, labels, dic_path, load=True):
        
        super(ImageSet, self).__init__()
        
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
        self.imgs = imgs
        self.num_data = len(self.imgs)
        self.labels = labels

        if load:
            self.label2id = self.load_pickle(dic_path)
        else:
            # if val:
            #     self.label2id = self.load_pickle(dic_path)
            #     counter = len(self.label2id)
            # else:
            self.label2id = {}
            counter = 0
            
            for label in self.labels.values():
                if label not in self.label2id:
                    self.label2id[label] = counter
                    counter += 1
            
            self.save_pickle(self.label2id, dic_path)
        
        self.num_labels = len(self.label2id)


    
    def __getitem__(self, index):
        root = '/Users/waynewu/10.backup_code/new_eyenet'
        path = os.path.join(root, self.imgs[index])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.label2id[self.labels[index]]
        return img, index, label
    
    def __len__(self):
        return len(self.imgs)
    
    # Load pickle files
    @staticmethod
    def load_pickle(file):
        with open(file,'rb') as f:
            return pickle.load(f)
    
    # save pickle files
    @staticmethod
    def save_pickle(array, file):
        with open(file,'wb') as f:
            return pickle.dump(array, f)

if __name__ == "__main__":

    data = ImageSet("./train_images.pkl", "./train_labels.pkl", "./label2id.pkl", load=False)
    dataloader = DataLoader(data, batch_size=64, shuffle=False)

    
    