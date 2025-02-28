import torch

model_0=torch.load("model_0_epoch.pth")
model_2=torch.load("model_2_epoch.pth")
model_10=torch.load("model_10_epoch.pth")

print(model_0['classifier.bias'])
print(model_2['classifier.bias'])
print(model_10['classifier.bias'])