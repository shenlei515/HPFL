from torch import Tensor 
from torch.utils.data.dataloader import DataLoader,Dataset
import torch
import numpy as np
print(sum([772, 1109, 150, 1200, 1006, 1137, 674, 1364, 1522, 1066]))
print(sum([612, 1289, 1047, 1666, 607, 1191, 743, 941, 1010, 894]))

sample_list=[(Tensor(5,),Tensor(3,)),(Tensor(5,),Tensor(3,)),(Tensor(5,),Tensor(3,))]
X, y=list(zip(*sample_list))
print(X)
print(y)
dataset=torch.utils.data.TensorDataset(torch.stack(X),torch.stack(y))
print(torch.stack(X).shape)
print(torch.stack(y).shape)
# dataset=torch.utils.data.TensorDataset(torch.concat(X),torch.concat(y))
print(torch.concat(X).shape)
print(torch.concat(y).shape)
test_data=torch.utils.data.dataloader.DataLoader(dataset, batch_size=128)

# mean=np.zeros((512,))
# var=np.ones((512,512))
# feature=np.ones((10000,512))
# print(np.random.multivariate_normal(mean, var, size=feature.shape))

mean=torch.zeros((512,))
var=torch.eye(512)
feature=torch.ones((10000,512))
m=torch.distributions.multivariate_normal.MultivariateNormal(mean, var)

print("m==",m.sample_n(10000).shape)
print(tuple(i for i in feature.size()))
print(torch.normal(2, 3, size=(10000,512)))
print(torch.normal(mean, var, size=(10000,512)))
