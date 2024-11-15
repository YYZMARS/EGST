import torch
import torch.nn as nn
import copy
from .cuda.emd_torch.emd import earth_mover_distance


def emd(template: torch.Tensor, source: torch.Tensor):

	emd_loss = torch.mean(earth_mover_distance(template, source))/(template.size()[1])
	return emd_loss


class EMDLosspy(nn.Module):
	def __init__(self):
		super(EMDLosspy, self).__init__()

	def forward(self, template, source):
		return emd(template, source)


if __name__ == '__main__':
	loss = EMDLosspy()
	a = torch.randn(4, 5, 3).numpy()
	b = copy.deepcopy(a)
	print(loss(a,b))


