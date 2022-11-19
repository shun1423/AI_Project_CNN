import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sn

from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Q1_model(nn.Module):
    def __init__(self):
        super(Q1_model, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.stage3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(800, 500, bias=True),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(500, 10, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = torch.flatten(out, 1)
        out = self.stage3(out)
        out = self.stage4(out)
        return out

##
train_dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomRotation(10),
                       transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2))
                   ]))
test_dataset =  datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
epochs = 30
batch = 32
net = Q1_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)


best_valid_acc = None
FILEPATH = './checkpoint'
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
        acc = (outputs.argmax(axis=1)==labels).float().sum()/batch
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc += acc.item()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if i % 10 == 9:
            running_loss = 0.0
            running_acc = 0.0
    epoch_loss = epoch_loss / (i+1)
    epoch_acc = epoch_acc /(i+1)
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

    valid_loss = valid_loss / (i+1)
    valid_acc = valid_acc / (i+1)
    print('[%d, %5d] TEST loss: %.3f acc: %.3f \n' %
          (epoch + 1, i + 1, valid_loss, valid_acc))
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_acc)

    if not best_valid_acc or valid_acc > best_valid_acc:
        print('### saving curent model...### \n')
        torch.save(net.state_dict(), FILEPATH)
        best_valid_acc = valid_acc


    f = open('./Q1_results.txt', 'a')
    f.write(
        "Epoch: %d | Train Loss: %.3f | Train Acc.: %.3f | Test Loss: %.3f | Test Acc.: %.3f \n" % (
            epoch+1, epoch_loss, epoch_acc, valid_loss, valid_acc)
    )
    f.close()

    epoch_loss = 0.0
    epoch_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0

visual('Q1', epoch_loss_list, valid_loss_list, epoch_acc_list, valid_acc_list)
