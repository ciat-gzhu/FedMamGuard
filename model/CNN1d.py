import torch
import torch.nn as nn
import torch.nn.functional as F


class Tabular1DCNN(nn.Module):
	def __init__(self, in_dim=58, out_dim=2, hidden=128):
		super(Tabular1DCNN, self).__init__()

		self.num_features = in_dim
		self.num_classes = out_dim

		# 卷积层块 1
		self.conv1 = nn.Sequential(
			nn.Conv1d(1, hidden // 4, 3, 1, 1),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True)
		)
		self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化层 (长度减半)

		# 卷积层块 2
		self.conv2 = nn.Sequential(
			nn.Conv1d(hidden // 4, hidden // 2, 3, 1, 1),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
		)
		self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化层 (长度再次减半)

		# 卷积层块 3 (可选，可以根据需要增加更多层)
		self.conv3 = nn.Sequential(
			nn.Conv1d(hidden // 2, hidden, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(hidden),
			nn.ReLU(inplace=True)
		)

		# 全连接层
		self.fc = nn.Linear(hidden, out_dim)

		print(f'1DCNN Param: {sum([p.numel() for p in self.parameters()])}')

	def forward(self, x):
		x = self.conv1(x.unsqueeze(1))
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.pool2(x)

		x = self.conv3(x)
		x = x.mean(dim=2)

		return self.fc(x)

