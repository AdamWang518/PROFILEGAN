# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import transforms
from model import discriminator, generator
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i] + 1)/2)  # 由于tanh是在-1 1 之间 要恢复道0 1 之间
        plt.axis("off")
    plt.show()
epochs = 200
lr = 0.0002
batch_size = 64
# 数据归一化
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(0.5, 0.5)  
])

train_set = datasets.MNIST('./', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = generator().to(device)
dis = discriminator().to(device)
g_optimizer = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))
loss_function = torch.nn.BCELoss()
for epoch in range(epochs):
    running_train_loss = 0.0
    running_test_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        dis.train()
        gen.train()
        size = images.size(0)#一個batch的size
        random_noise = torch.randn(size, 128, device=device)
        # print(size)
        d_optimizer.zero_grad()  # 将上述步骤的梯度归零
        real_output = dis(images)  # 对判别器输入真实的图片，real_output是对真实图片的预测结果
        d_real_loss = loss_function(real_output,
                                    torch.ones_like(real_output)
                                    )
        d_real_loss.backward() #求解梯度，訓練辨識真物
        gen_img = gen(random_noise)
        fake_output=dis(gen_img.detach())
        d_fake_loss= loss_function(fake_output,
                                    torch.zeros_like(fake_output)
                                    )
        d_fake_loss.backward()#求解梯度，訓練辨識偽物
        d_loss = d_real_loss + d_fake_loss
        d_optimizer.step()  # 优化
         # 得到生成器的损失
        g_optimizer.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_function(fake_output,
                               torch.ones_like(fake_output))
        # 求與真物的差距
        g_loss.backward()
        g_optimizer.step()
        running_train_loss+=g_loss
    print(f"Epoch [{epoch+1}/{epochs}], train_Loss: {running_train_loss/len(train_loader)}")
    if (epoch + 1) % 5 == 0:
        torch.save(dis, f'Discriminator_epoch_{epoch}.pth')
        torch.save(gen, f'Generator_epoch_{epoch}.pth')
        print('Model saved.')





