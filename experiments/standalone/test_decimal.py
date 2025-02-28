import numpy as np
import decimal
import torch

print(sum([6630,3448,6186,3102,1060,6377,8199,2853,5184,6961]))

a=torch.Tensor([[0,1],
                [0,1]])
a=torch.softmax(a, dim=-1)
print(a)
a=np.mean(a.numpy(), axis=-1)
print(a)
print(sum([0.0099984 ,0.08025648, 0.08159163, 0.08724197, 0.09308684, 0.0945752,
 0.10550435, 0.10617466, 0.11923684, 0.18135315]))

decimal.getcontext().prec=500
a=np.array([0.0000000001,0.555555555555555555555555,0.01*1e-300])
a=a.astype(str)
print(a)
a=a.astype(decimal.Decimal)
a=a*decimal.Decimal(1e-300)

print(a)