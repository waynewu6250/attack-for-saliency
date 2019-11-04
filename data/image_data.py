import torch as t
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import tqdm
import pickle

class ImageSet(Dataset):

    def __init__(self, img_path, label_path):
        
        super(ImageSet, self).__init__()
        
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
        self.imgs = self.load_pickle(img_path)
        self.labels = self.load_pickle(label_path)
        self.label2id = {}
        counter = 0
        for label in self.labels.values():
            if label not in self.label2id:
                self.label2id[label] = counter
                counter += 1
        self.num_labels = len(self.label2id)
    
    def __getitem__(self, index):
        root = '/Users/waynewu/10.backup_code/new_eyenet'
        path = os.path.join(root, self.imgs[index])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.label2id[self.labels[self.imgs[index]]]
        return img, index, label
    
    def __len__(self):
        return len(self.imgs)
    
    # Load pickle files
    @staticmethod
    def load_pickle(file):
        with open(file,'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":

    data = ImageSet("./train_images.pkl", "./train_labels.pkl")
    dataloader = DataLoader(data, batch_size=64, shuffle=False)
    for img, index, label in dataloader:
        print(img[0][0])
        print(index[0])
        print(label[0])
        break

    
    