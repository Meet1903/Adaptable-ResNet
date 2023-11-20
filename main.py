import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from resnet import *
import os
import argparse
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_gradients(model):
    return sum(p.grad.numel() for p in model.parameters() if p.grad is not None)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_data_load_time = 0
    total_run_time = 0
    total_train_time = 0
    #below line is for Run time start
    run_start_time = time.perf_counter()
    #below line is for Data-loading time start
    data_load_start_time = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #below line is for Data-loading time end
        data_load_end_time = time.perf_counter()
        total_data_load_time += (data_load_end_time - data_load_start_time)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        #below line is for Train time start
        train_start_time = time.perf_counter()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        #below line is for Train time end
        train_end_time = time.perf_counter()
        total_train_time += (train_end_time - train_start_time)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # print("Epoch: ", epoch, batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #below line is for for Data-loading time start
        data_load_start_time = time.perf_counter()

    run_end_time = time.perf_counter()
    data_loading_time_per_epoch.append(total_data_load_time)
    training_time_per_epoch.append(total_train_time)
    total_run_time_per_epoch.append(run_end_time - run_start_time)
    print("\nAfter Epoch: %d Loss: %.3f | Acc: %.3f%% (%d/%d)"% (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total) )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # print("Epoch: ", epoch, batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--datapath', default='./data', type=str)
    args = parser.parse_args()
    print("Device= ",args.device, "Workers = ", args.workers, "Optimizer= ", args.optimizer)
    if args.device.lower() == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.datapath, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root=args.datapath, train=False, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False, num_workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    best_acc = 0  # best test accuracy
    start_epoch = 0 
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    # Uncomment below line for Resnet with no Normalization. Also comment above line.
    # net = ResNetNoNorm(BasicBlockNoNorm, [2, 2, 2, 2])
    net = net.to(device)
    if device == 'cuda':
        print("Device = CUDA")
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    else:
        print("Device: CPU")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,  momentum=0.9, weight_decay=5e-4)
    elif args.optimizer.lower() == 'nesterov':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,  momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    data_loading_time_per_epoch = []
    training_time_per_epoch = []
    total_run_time_per_epoch = []

    
    for epoch in range(start_epoch, start_epoch+1):
        train(epoch)
        test(epoch)
        scheduler.step()
        print("\nData loading time: ", data_loading_time_per_epoch[epoch], "\nTraining time: ", training_time_per_epoch[epoch], "\nRun time: ", total_run_time_per_epoch[epoch])
    print("Total Data Loading time: ", sum(data_loading_time_per_epoch))
    print("Total run time: ", sum(total_run_time_per_epoch))
    print("Total training time: ", sum(training_time_per_epoch))

    print("Total Trainable Parameters: ", count_parameters(net))
    print("Total Gradients: ", count_gradients(net))