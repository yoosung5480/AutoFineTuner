import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision
import time
import os
import matplotlib.pyplot as plt
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
print('STEP 1: LOADING DATASET')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = dsets.CIFAR10(root='./data/CIFAR10/',
                            train=True,
                            transform=transform_train,
                            download=True)

test_dataset = dsets.CIFAR10(root='./data/CIFAR10/',
                           train=False,
                           transform=transform_test)

# reducing the dataset
reduced_train_dataset = []
for images, labels in train_dataset:
    if labels < 3:
        reduced_train_dataset.append((images, labels))

reduced_test_dataset = []
for images, labels in test_dataset:
    if labels < 3:
        reduced_test_dataset.append((images, labels))

print("The number of training images : ", len(reduced_train_dataset))
print("The number of test images : ", len(reduced_test_dataset))


print('STEP 2: MAKING DATASET ITERABLE')

train_loader = torch.utils.data.DataLoader(dataset=reduced_train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=reduced_test_dataset,
                                          batch_size=100,
                                          shuffle=False)

class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print('STEP 3: CREATE MODEL CLASS (VGG16)')
#############
# CODE HERE # 
#############
# 목표!
# VGGNET을 구현해야한다!
# padding = 1, stride = 1 로 고정이라고한다.
# 그런데 우리 데이터는 크기가 [3, 32, 32,] # 데이터는 15000 장이다.
#### 이미 텐서가 작은 상태니깐, 
## 224 = (2^6)*
## 32 = 2^6 --> 
# 원래 논문은 [3, 224, 224] -> [64, 224, 224 ] -> [128, 112, 112] -> [256, 56, 56] -> [512, 28, 28] -> [512, 14, 14] -> [512, 7, 7]
## 내 코드에서는..
#  [3, 32, 32] -> [64, 16, 16 ] -> [128, 8, 8] -> [256, 4, 4] -> [512, 2, 2] -> [512, 1, 1] -> [512, 1, 1]
## 입력텐서 하드코딩하지말고, 그냥 변수화해서 32의 배수면 다 처리가능케해보자.
class VGGLayer(nn.Module):
    def __init__(self, b : int, c : int, w : int , h : int, cfg : int, num_layer:int):
        '''
        b : batch 크기
        c : 채널 (RGB면, 3)
        w : 이미지 넓이
        h : 이미지 높이

        cfg : 변환되는 featuremap 개수
        num_layer : 레이어개수

        합성곱층 크기 ( 3 * 3* 3 )
        w == h == 32의 배수여야함!.. 
        '''
        self.layers = []
        for i in range(num_layer):
            self.layers.append([
                nn.Conv2d(c, cfg, kernel_size=3, stride=1, padding=1),  
                nn.ReLU()
            ])

##### 핵심!
## kernel
# stride = 1, padding = 1로 고정
# conv 커널크기는 =  [3, 3, 3]

## pooling
# stride = 2, padding = 0 
# 풀링 커널크기는 = [2, 2]    


## 매 MP 마다 형태가 2배씩 줄어든다.
# 그런데 MP가 4개밖에 없다. 논문에서는 16개의 레이어에서 총 5개의 pooling이 있다.
# cfg = [64, 64, 'MP', 128, 128, 128, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP']       # 원본
cfg = [ 64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP']

class VGG(nn.Module):
    def __init__(self, num_layer = 1):
        super(VGG, self).__init__()
        self.VGG16 = self._make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self._make_classifier(num_layer)

    def forward(self, x):
        out = self.VGG16(x)                  # [B, 512, 1, 1]
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)      # [B, 512]
        out = self.classifier(out)           # [B, 3]
        return out
    
    def _make_classifier(self, num_layer : int):
        layer = []
        for i in range(num_layer):
            layer += [nn.Linear(512, 512),
                      nn.ReLU(inplace=True),
                      ]
        layer += [nn.Linear(512, 3)]
        return nn.Sequential(*layer)


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'MP':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)
    

print('STEP 4: INSTANTIATE MODEL CLASS')

model = VGG()
num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
#######################
#  USE GPU FOR MODEL  #
#######################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


print('STEP 5: INSTANTIATE LOSS CLASS')

criterion = nn.CrossEntropyLoss()

print('STEP 6: INSTANTIATE OPTIMIZER CLASS')

learning_rate = 1e-2
momentum = 0.9
weight_decay = 5e-4

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum, weight_decay = weight_decay)


import time
print('STEP 7: TRAIN THE MODEL')
num_epochs = 1

#############
# CODE HERE #
#############
for epoch in range(num_epochs):
    start = time.time()
    model.train()       # dropout, batchNore 레이어 동작 전환. 가중치 업뎃자체는 optimizer가함.
    running_loss = 0.0
    correct = 0
    total = 0
    
    #-----------------------------------------실제 훈련부분-----------------------------------------#
    for imgs, labels in train_loader:
        # GPU로 보내기
        imgs, labels = imgs.to(device), labels.to(device)

        # 옵티마이저 가중치 초기화
        optimizer.zero_grad()

        # !!!! 실제로 훈련된 모델에 예측값 받아보기 !!!!
        outputs = model(imgs)
        # 실제 예측값과 정답값 차이 계산
        loss = criterion(outputs, labels)
        loss.backward()     # gradient 계산 (오차 역전파)
        optimizer.step()    #  전달

        # 통계
        running_loss += loss.item() * imgs.size(0)    # 맨 앞쪽은 BATCH_SIZE이니깐, 각 객체들의 로스를 더한거
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    #----------------------------------------------------------------------------------#
    # 평균 손실, 정확도 계산
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    elapsed_time = time.time() - start

    ##################
    #   Print Log    #
    ##################
    print(f"Epochs: {epoch}. "
            f"Loss: {epoch_loss:.4f}. "
            f"Accuracy: {epoch_acc:.2f}. "
            f"Elapsed time: {elapsed_time:.2f} sec")

