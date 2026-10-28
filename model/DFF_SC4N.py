import torch
import torch.nn as nn


class DFF_SC4N(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super(DFF_SC4N, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, out_dim)
        )

    def forward(self, x):
        gru_out = self.gru(x.unsqueeze(1))[0]
        x = self.fc(gru_out[:, -1, :])
        return x


if __name__ == '__main__':
    with torch.no_grad():
        input_dim = 58
        model = DFF_SC4N(input_dim, 8, 128)
        model.eval()

        dummy_input = torch.randn(32, input_dim)
        output = model(dummy_input)
        print(output.size())
