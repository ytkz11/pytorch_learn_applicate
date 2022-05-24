#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 5/24/2022 1:08 PM 
# @Author : DKY
# @File : uilze_pt.py

# utilize pt file to predict


from mnist1 import CNN
import torch
import torchvision.datasets
from torch.autograd import Variable
def   main():
    PATH = r'my_model.pth'
    cnnmodel = torch.load(PATH)
    cnnmodel.eval()
    train_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000] / 255
    test_y = test_data.test_labels[:2000]
    test_output = cnnmodel(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real numpy')

if __name__ == '__main__':
    main()
a = 0
