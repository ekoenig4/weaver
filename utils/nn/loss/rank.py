import torch

class PointwiseLoss(torch.nn.MSELoss):
    def forward(self, input, target, batch):
        return super().forward(input, target)

class PairwiseLoss(torch.nn.MarginRankingLoss):
    def forward(self, input, target, batch):
        n_elements = batch.unique(return_counts=True)[1]
        offset = n_elements.cumsum(0)-n_elements[0]
        repeats = torch.repeat_interleave(n_elements, n_elements)
        arange = torch.arange(len(batch)).to(input.device)
        rows = torch.repeat_interleave(arange, repeats)

        n_elements = n_elements[batch]
        offset = n_elements.cumsum(dim=0)-n_elements[0]
        repeats = torch.repeat_interleave(offset-n_elements*batch, n_elements)
        arange = torch.arange( len(rows) ).to(input.device)
        cols = arange - repeats

        targ_diff = (target[rows]-target[cols]).sign()

        rows = rows[targ_diff != 0]
        cols = cols[targ_diff != 0]

        x1, x2, y = input[rows], input[cols], targ_diff[targ_diff != 0]
        return super().forward(x1, x2, y)

from utils.nn.scatter_tools import scatter_mean, scatter_sum, scatter_randperm, scatter_sort, scatter_cumsum_inv

class ListwiseLoss(torch.nn.Module):
    def forward(self, input, target, batch, eps=1e-12):
        target_shuffle, randperm = scatter_randperm(target, batch)
        input_shuffle = input[randperm]

        target_sorted, sortarg = scatter_sort(target_shuffle, batch, descending=True)
        input_sorted = input_shuffle[sortarg]

        cumsum = scatter_cumsum_inv(input_sorted.exp(), batch)
        loss = torch.log(cumsum + eps) - input_sorted
        loss = scatter_sum(loss, batch)

        # cumsum = scatter_cumsum_inv(target_sorted.exp(), batch)
        # minim = torch.log(cumsum + eps) - target_sorted
        # minim = scatter_mean(minim, batch)
        # return (loss - minim).mean()
        return loss.mean()
