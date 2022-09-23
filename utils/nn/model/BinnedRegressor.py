import torch

class BinnedRegressor(torch.nn.Module):
    def __init__(self, n_in=1, layers=[32,64,128], bins=None):
        super().__init__()

        self.register_buffer('bins', bins)

        layers = [n_in] + layers
        self.mlp = torch.nn.Sequential()
        for _n_in, _n_out in zip(layers[:-1],layers[1:]):
            self.mlp.append(
                torch.nn.Sequential(
                    torch.nn.Linear(_n_in, _n_out),
                    torch.nn.ReLU()
                )
            )

        self.mlp.append(torch.nn.Sequential(
            torch.nn.Linear(_n_out, len(bins)-1),
            torch.nn.Softmax(dim=-1)
        ))

    def forward(self, x):
        return self.mlp(x)