import torch
import torchvision as tv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

from model import AlexNet, Base_model, Model_1, Model_2, Model_3
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data Preperation
face_dataset_path = './face_dataset'

class Train_Face(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.save_path = path
        self.data_path = os.path.join(path, "facescrub_train")
        if not os.path.isfile(os.path.join(self.save_path, 'train_set.txt')):
            self.generate_label()
        self.create_index()

    def __getitem__(self, idx):
        img_dir = self.img_list[idx]
        # img = Image.open(img_dir)
        img = cv2.imread(img_dir)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.label_list[idx]

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return len(self.img_list)

    def create_index(self):
        self.img_list = []
        self.label_name_list = []

        listfile = os.path.join(self.save_path, 'train_set.txt')

        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(l[0])
                self.label_name_list.append(l[1])

        self.Name2Idx()


    def Name2Idx(self):
        self.label_list = []
        name_list = list(set(self.label_name_list))
        name_dict = {string: i for i, string in enumerate(name_list)}
        for name in self.label_name_list:
            self.label_list.append(name_dict.get(name))


    def generate_label(self):
        face_name = os.listdir(self.data_path)
        face_name = [folder for folder in face_name]

        for name in face_name:
            file_root = os.path.join(self.data_path, name)
            file_list = os.listdir(file_root)
            file_list = [file for file in file_list]
            for file in file_list:
                image_dir = os.path.join(file_root, file)
                f = open(os.path.join(self.save_path,'train_set.txt'), 'a')
                f.write(image_dir + ' ' + name + '\n')
                f.close()


class Test_Face(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.save_path = path
        self.data_path = os.path.join(path, "facescrub_test")
        if not os.path.isfile(os.path.join(self.save_path, 'test_set.txt')):
            self.generate_label()
        self.create_index()

    def __getitem__(self, idx):
        img_dir = self.img_list[idx]
        # img = Image.open(img_dir)
        img = cv2.imread(img_dir)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.label_list[idx]

        if transform is not None:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return len(self.img_list)

    def create_index(self):
        self.img_list = []
        self.label_name_list = []

        listfile = os.path.join(self.save_path, 'test_set.txt')

        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(l[0])
                self.label_name_list.append(l[1])

        self.Name2Idx()


    def Name2Idx(self):
        self.label_list = []
        name_list = list(set(self.label_name_list))
        name_dict = {string: i for i, string in enumerate(name_list)}
        for name in self.label_name_list:
            self.label_list.append(name_dict.get(name))


    def generate_label(self):
        face_name = os.listdir(self.data_path)
        face_name = [folder for folder in face_name]

        for name in face_name:
            file_root = os.path.join(self.data_path, name)
            file_list = os.listdir(file_root)
            file_list = [file for file in file_list]
            for file in file_list:
                image_dir = os.path.join(file_root, file)
                f = open(os.path.join(self.save_path,'test_set.txt'), 'a')
                f.write(image_dir + ' ' + name + '\n')
                f.close()


## DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
     transforms.Resize([32,32])])

train_dataset = Train_Face(face_dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = Test_Face(face_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    epochs = 200
    batch = 16
    net = Model_1().to(device)
    model_name = 'Model_1'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

    epoch_loss_list = []
    epoch_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        epoch_loss = 0.0
        epoch_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(axis=1) == labels).float().sum() / batch
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += acc.item()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if i % 10 == 9:
                running_loss = 0.0
                running_acc = 0.0

        epoch_loss = epoch_loss / (i + 1)
        epoch_acc = epoch_acc / (i + 1)
        print('[%d, %5d] FINISH loss: %.3f acc: %.3f' %
              (epoch + 1, i + 1, epoch_loss, epoch_acc))
        epoch_loss_list.append(epoch_loss)
        epoch_acc_list.append(epoch_acc)

        net.eval()
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(axis=1) == labels).float().sum() / batch
            valid_loss += loss.item()
            valid_acc += acc.item()

        valid_loss = valid_loss / (i + 1)
        valid_acc = valid_acc / (i + 1)
        print('[%d, %5d] TEST loss: %.3f acc: %.3f \n' %
              (epoch + 1, i + 1, valid_loss, valid_acc))
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        f = open('./Q3_'+ model_name+ '_results.txt', 'a')
        f.write(
            "Epoch: %d | Train Loss: %.3f | Train Acc.: %.3f | Test Loss: %.3f | Test Acc.: %.3f \n" % (
                epoch, epoch_loss, epoch_acc, valid_loss, valid_acc)
        )
        f.close()

        if epoch % 50 == 0:
            ep = str(epoch)
            visual('Q3_'+model_name+'_'+ep, epoch_loss_list, valid_loss_list, epoch_acc_list, valid_acc_list)

if __name__ == '__main__':
    main()

