import torch
import torch.nn as nn
import torch.nn.functional as F


class CFL_IDS(nn.Module):
	def __init__(self, in_dim, out_dim, hidden=64):
		super(CFL_IDS, self).__init__()

		# CNN layers
		self.cnn1 = nn.Sequential(
			nn.Conv1d(1, hidden, 2, padding='same'),
			nn.ReLU(inplace=True),
		)
		self.cnn2 = nn.Sequential(
			nn.Conv1d(1, hidden, 4, padding='same'),
			nn.ReLU(inplace=True),
		)
		self.cnn3 = nn.Sequential(
			nn.Conv1d(1, hidden, 6, padding='same'),
			nn.ReLU(inplace=True),
		)

		self.gru = nn.GRU(hidden, hidden, batch_first=True)

		self.fc = nn.Sequential(
			nn.Linear(hidden, hidden * 2),
			nn.ReLU(inplace=True),
			nn.Linear(hidden * 2, out_dim)
		)

	def forward(self, x):
		x = x.unsqueeze(1)

		x1 = self.cnn1(x).transpose(1, 2)
		x2 = self.cnn2(x).transpose(1, 2)
		x3 = self.cnn3(x).transpose(1, 2)
		x = torch.cat((x1, x2, x3), dim=1)
		gru_out = self.gru(x)[0]

		return self.fc(gru_out[:, -1, :])


if __name__ == '__main__':
	with torch.no_grad():
		input_dim = 58
		model = CFL_IDS(input_dim, 8, 64)
		model.eval()

		dummy_input = torch.randn(32, input_dim)
		output = model(dummy_input)
		print(output.size())
