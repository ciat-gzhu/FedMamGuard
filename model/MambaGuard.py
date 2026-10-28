import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from mamba_ssm.modules.mamba2 import Mamba2

from HFCM import HFCM, HFCM_NoAttn, PermuteRMSNorm


class BiGRU(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden//2, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class MambaGuard(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, simam=True, hfcm=True, seq_model='mamba', residual=False):
        super(MambaGuard, self).__init__()
        # 对输入shape=(N, H)做两层2x降采样, 得到shape=(N, H//4, hidden)的序列
        if hfcm:
            if simam:
                self.hfcm = HFCM(hidden)
            else:
                self.hfcm = HFCM_NoAttn(hidden)
        else:
            self.hfcm = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=3, stride=1),
                nn.SiLU(),
                PermuteRMSNorm(hidden//2),
            )
        # 在尾部连接[CLS]
        self.cls_token = nn.Parameter(torch.zeros((1, 1, hidden)))
        if seq_model == 'mamba':
            # 做一层Mamba
            self.seq_model = Mamba2(d_model=hidden, d_state=32, d_conv=4, headdim=32, expand=2)
        elif seq_model == 'lin':
            from linformer import Linformer
            self.seq_model = Linformer(
                hidden, seq_len=in_dim//4+1,
                depth=1, heads=hidden//32,
                reversible=True, dropout=0.1
            )
        elif seq_model == 'gru':
            self.seq_model = BiGRU(hidden)
        else:
            self.seq_model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hidden, nhead=hidden//32,
                    dim_feedforward=hidden*4,
                    batch_first=True
                ), num_layers=1, norm=nn.LayerNorm(hidden),
                mask_check=False,
            )
        # 对[CLS]分类
        self.output_layer = nn.Linear(hidden, out_dim)
        self.residual = residual

        print(f'{seq_model.upper()} Param: {sum([p.numel() for p in self.parameters()])}')

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.hfcm(x).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        if self.residual:
            x_res = x

        x = self.seq_model(x)
        if self.residual:
            x = x + x_res

        x = x[:, -1, :]
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    in_dim = 58
    out_dim = 8
    hidden = 128
    device = torch.device('cuda')

    data = torch.randn((1, in_dim), device=device)

    from thop import profile

    @torch.no_grad()
    def calc_flops(model):
        # 计算FLOPS和参数量
        model.eval()
        try:
            flops, params = profile(model, inputs=(data,))

            # 打印结果
            print(f"FLOPS: {flops/1e6:.3f} GFLOPS")
            print(f"Params: {params} params")

        except Exception as e:
            print(f"计算过程中出现错误: {str(e)}")
            print("可能原因：模型包含未被thop支持的自定义层")

    # model = MambaGuard(in_dim, out_dim, hidden, seq_model='mamba', simam=True, hfcm=True)
    # model = MambaGuard(in_dim, out_dim, hidden, seq_model='mamba', simam=False, hfcm=True)
    # model = MambaGuard(in_dim, out_dim, hidden, seq_model='mamba', hfcm=False)
    # seq_model = 'mamba'
    for seq_model in ['mamba', 'lin', 'gru', 'transformer']:
        model = MambaGuard(in_dim, out_dim, hidden, seq_model=seq_model).to(device)
        calc_flops(model)
        # with torch.no_grad():
        #     out = model(data)
        #     print(seq_model, out.size())
    from MLP import MLP
    model = MLP(in_dim, out_dim, hidden).to(device)
    calc_flops(model)
    from CNN1d import Tabular1DCNN
    model = Tabular1DCNN(58, 8, 128).to(device)
    calc_flops(model)
