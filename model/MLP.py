import torch.nn as nn


class MLP(nn.Module):
	def __init__(self, in_dim, out_dim, hidden):
		super(MLP, self).__init__()
		# 对输入shape=(N, H)做两层2x降采样, 得到shape=(N, H//4, hidden)的序列
		self.mlp = nn.Sequential(
			nn.Linear(in_dim, hidden),
			nn.BatchNorm1d(hidden),
			nn.ReLU(inplace=True),
			nn.Linear(hidden, out_dim)
		)

		print(f'MLP Param: {sum([p.numel() for p in self.parameters()])}')

	def forward(self, x):
		return self.mlp(x)
