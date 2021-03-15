import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import StepLR
import resnet
import os
import math
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test(model):
    model.eval()
    test_acc = 0
    for img, label in test_dataLoader:
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        out = out[0]
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        test_acc += num_correct.data.item()
    return test_acc/len(test_data)

##inter-correlation
def correlation(feature):
    CR = []
    for i in range(feature.size(0)):
        tmp = feature[i].reshape(1,feature[i].size(0), feature[i].size(1), feature[i].size(2))
        tmp = tmp.expand(feature.size(0), -1, -1, -1)
        cos_sim = (tmp*feature).sum(dim=[1,2,3]) / (tmp**2).sum(dim=[1,2,3]).sqrt() / (feature**2).sum(dim=[1,2,3]).sqrt()
        CR.append(cos_sim)
    CR = torch.stack(CR)
    return CR

def correlation_logits(feature):
    CR = []
    for i in range(feature.size(0)):
        tmp = feature[i].reshape(1,feature[i].size(0))
        tmp = tmp.expand(feature.size(0), -1)
        cos_sim = (tmp*feature).sum(dim=-1) / (tmp**2).sum(dim=-1).sqrt() / (feature**2).sum(dim=-1).sqrt()
        CR.append(cos_sim)
    CR = torch.stack(CR)
    return CR

##group features into different cluster
def cluster(feature, label, num_classes):
    length = feature.size(0)
    center_list = []
    res = []
    valid_center_list = []
    for i in range(num_classes):
        center_list.append([])
    for i in range(length):
        center_list[label[i]].append(feature[i])
    for i in range(num_classes):
        if(len(center_list[i]) > 0):
            center_list[i] = sum(center_list[i]) / len(center_list[i])
            valid_center_list.append(center_list[i])
    for i in range(length):
        res.append(center_list[label[i]])
    
    return torch.stack(res), torch.stack(valid_center_list)

def cluster_filter(feature, pred, label, num_classes, mode='intra'):  
    length = feature.size(0)
    center_list = []
    res = []
    valid_center_list = []
    for i in range(num_classes):
        center_list.append([])
    for i in range(length):
        if(pred[i] == label[i]):
            center_list[label[i]].append(feature[i])
    for i in range(num_classes):
        if(len(center_list[i]) > 0):
            center_list[i] = sum(center_list[i]) / len(center_list[i])
            valid_center_list.append(center_list[i]) 
    for i in range(length):
        if(len(center_list[label[i]]) > 0):
            res.append(center_list[label[i]])
        else:
            res.append(feature[i])
    if mode == 'intra':
        return torch.stack(res)
    elif(len(valid_center_list) > 0):
        return torch.stack(valid_center_list)
    else:
        return []



##hyper-parameters settings
batch_size = 128
lr = 0.1
num_epochs = 200

dataset = 'cifar100'
t_name = 'resnet101'
s_name = 'resnet18'

t_path = 'saved_model/' + dataset + '_' + t_name + '.pkl'
s_path = 'saved_model/S' + s_name[-2:] + '_T' + t_name[-3:] + '_' + dataset + '.pkl'
log_path = 'train_log.txt'
curve_path = 'saved_model/train_curve.jpg'
print(t_path)
print(s_path)
Teacher = torch.load(t_path)
##load data
train_process = transforms.Compose([transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(), 
                              transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
test_process = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
if dataset == 'cifar100':
    train_data = datasets.CIFAR100(root='data/cifar100', transform=train_process, train=True, download=True)
    test_data = datasets.CIFAR100(root='data/cifar100', transform=test_process, train=False, download=True)
    num_classes = 100
elif dataset == 'cifar10':
    train_data = datasets.CIFAR10(root='data/cifar10', transform=train_process, train=True, download=True)
    test_data = datasets.CIFAR10(root='data/cifar10', transform=test_process, train=False, download=True)
    num_classes = 10
train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False)



##load model
if(s_name == 'resnet18'):
    Student = resnet.resnet18(num_classes=num_classes)
    Student.fc = nn.Linear(128, num_classes)
    Student.branch1 = resnet.conv1x1(16, 128)
    Student.branch2 = resnet.conv1x1(32, 256)
    Student.branch3 = resnet.conv1x1(64, 512)
    Student.branch4 = resnet.conv1x1(128, 1024)
elif(s_name == 'resnet34'):
    Student = resnet.resnet34(num_classes=num_classes)
    Student.fc = nn.Linear(128, num_classes)
    Student.branch1 = resnet.conv1x1(16, 128)
    Student.branch2 = resnet.conv1x1(32, 256)
    Student.branch3 = resnet.conv1x1(64, 512)
    Student.branch4 = resnet.conv1x1(128, 1024)
elif(s_name == 'resnet50'):
    Student = resnet.resnet50(num_classes=num_classes)
    Student.fc = nn.Linear(512, num_classes)
    Student.branch1 = resnet.conv1x1(64, 128)
    Student.branch2 = resnet.conv1x1(128, 256)
    Student.branch3 = resnet.conv1x1(256, 512)
    Student.branch4 = resnet.conv1x1(512, 1024)

print('#params of Student:', sum([para.numel() for para in Student.parameters()]))
for para in Teacher.parameters():
    para.requires_grad = False
    
##transfer to GPU
Teacher = Teacher.cuda()
Student = Student.cuda()

CE = nn.CrossEntropyLoss()
MSE = nn.MSELoss()

CE = CE.cuda()
MSE = MSE.cuda()

loss_list = []
train_acc_list = []
test_acc_list = []

#initialize center list
center_list = []
for i in range(num_classes):
    center_list.append([])

use_filter = True
    
for epoch in range(num_epochs):
    Student.train()
    if epoch < 80:
        optimizer = optim.SGD(Student.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    elif epoch < 120:
        optimizer = optim.SGD(Student.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(Student.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
       
    cost = 0.0
    cnt = 0
    num_correct = 0
    s_tick = time.time()

    for i, (img, label) in enumerate(train_dataLoader):
        s = time.time()
        img, label = img.cuda(), label.cuda()
        t_out, tf1, tf2, tf3, tf4 = Teacher(img)
        s_out, sf1, sf2, sf3, sf4 = Student(img)

        _, t_pred = torch.max(t_out, 1)
        _, s_pred = torch.max(s_out, 1)
        if use_filter:
            ## in most cases, feature map in penultimate layer and logits are useful
            TF4 = cluster_filter(tf4, t_pred, label, num_classes)
            SF4 = cluster_filter(sf4, s_pred, label, num_classes)
            T_OUT = cluster_filter(t_out, t_pred, label, num_classes)
            S_OUT = cluster_filter(s_out, s_pred, label, num_classes)
            
            
            
            _, T_center_list = cluster(tf4, label, num_classes)
            _, S_center_list = cluster(sf4, label, num_classes)
            T_OUT, T_logits = cluster(t_out, label, num_classes)
            S_OUT, S_logits = cluster(s_out, label, num_classes)
            
        else:
            TF4, T_center_list = cluster(tf4, label, num_classes)
            SF4, S_center_list = cluster(sf4, label, num_classes)
            T_OUT, T_logits = cluster(t_out, label, num_classes)
            S_OUT, S_logits = cluster(s_out, label, num_classes)

        
        center, _ = cluster(s_out, label, num_classes)
        
        intra_cor = 0.002 * (MSE((tf4-TF4).detach(), sf4-SF4.detach()) / TF4.size(1) / TF4.size(2) / TF4.size(3) + \
                MSE((t_out-T_OUT).detach(), s_out-S_OUT.detach()) / T_OUT.size(1)) + 0.1 * MSE(s_out, center.detach())
        
        T_cor = correlation(T_center_list)
        S_cor = correlation(S_center_list)
        
        T_cor_logits = correlation_logits(T_logits)
        S_cor_logits = correlation_logits(S_logits)
        inter_cor = 0.04 * (MSE(T_cor.detach(), S_cor) / T_cor.size(1) + MSE(T_cor_logits.detach(), S_cor_logits) / T_cor_logits.size(1))
        
        category_loss = intra_cor + inter_cor
        
        cross_entropy = CE(s_out, label)
        loss_KD = 0.00001 * MSE(t_out.detach(), s_out)

        
        alpha = 0.9
        loss = alpha * cross_entropy + category_loss + (1 - alpha) * loss_KD

        _, pred = torch.max(s_out, 1)
    
    
        num_correct += (pred == label).sum().item()
        cost += loss.item()
        cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e = time.time()
        if(i%5 == 0):
          print('epoch %d: %d/%d, time:%.2fs  ce:%.3f intra:%.3f inter:%.3f category_loss:%.3f kd:%.3f'%(epoch, cnt, math.ceil(len(train_data)/batch_size), e-s, cross_entropy.item(), intra_cor.item(),inter_cor.item(), category_loss.item(),loss_KD.item()), end='\r')
    e_tick = time.time()
    loss_list.append(cost/cnt)
    train_acc_list.append(num_correct/len(train_data))
    test_acc_list.append(test(Student))
    log = 'epoch:%d loss:%f train_acc:%f test_acc:%f time:%.2fs'%(epoch, cost/cnt, num_correct/len(train_data), test(Student), e_tick-s_tick)
    print(log)
    open(log_path, 'a').write(log)
    torch.save(Student.state_dict(), s_path)
print('Maximum test accuracy:', max(test_acc_list))


x_axis = range(num_epochs)
plt.figure(figsize=(10,13))
plt.subplot(3,1,1)
plt.plot(x_axis, loss_list, 'r-')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(3,1,2)
plt.plot(x_axis, train_acc_list, 'g-')
plt.xlabel('epoch')
plt.ylabel('train accuracy')

plt.subplot(3,1,3)
plt.plot(x_axis, test_acc_list, 'b-')
plt.xlabel('epoch')
plt.ylabel('test accuracy')

plt.savefig(curve_path, dpi=600)
plt.show()