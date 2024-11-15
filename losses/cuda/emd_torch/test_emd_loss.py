import torch
import numpy as np
import time
from emd import earth_mover_distance



# emd
p1 = torch.from_numpy(np.array([[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=np.float32)).cuda()
p1 = p1.repeat(3, 1, 1)
p2 = torch.from_numpy(np.array([[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=np.float32)).cuda()
p2 = p2.repeat(3, 1, 1)


d = earth_mover_distance(p1, p2, transpose=False)
print(d)

loss = d[0] / 2 + d[1] * 2 + d[2] / 3
print(loss)


