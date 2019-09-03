import torch
import torch.nn.functional as F
import numpy as np


M = torch.tensor(np.ones((10,10))) / 10
print(M)


loss = F.l1_loss(torch.tensor(np.ones((10,10))),M)

torch.optim.Adam(M.parameters(),lr=0.01)


loss.backward()
