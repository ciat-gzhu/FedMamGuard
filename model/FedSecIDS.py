import torch
import torch.nn as nn
import torch.nn.functional as F


class FedSecIDS(nn.Module):
	def __init__(self, in_dim, out_dim, hidden=64):
		super(FedSecIDS, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.Conv1d(1, hidden // 2, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(2),
		)

		self.cnn2 = nn.Sequential(
			nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(2),
		)
		self.dropout = nn.Dropout(0.25)

		self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

		# 全连接层 (MLP)
		self.fc = nn.Sequential(
			nn.Linear(hidden, hidden*2),
			nn.ReLU(inplace=True),
			nn.Linear(hidden*2, out_dim)
		)

	def forward(self, x):
		x_cnn = self.cnn1(x.unsqueeze(1))
		x_cnn = self.cnn2(x_cnn).transpose(1, 2)

		x_cnn = self.dropout(x_cnn)  # 根据图 5，在展平后应用 dropout

		lstm_output = self.lstm(x_cnn)[0]
		# 使用最后一个时间步的输出用于 MLP
		return self.fc(lstm_output[:, -1, :])


if __name__ == '__main__':
	with torch.no_grad():
		input_dim = 58
		model = FedSecIDS(input_dim, 8, 128)
		model.eval()

		dummy_input = torch.randn(32, input_dim)
		output = model(dummy_input)
		print(output.size())
