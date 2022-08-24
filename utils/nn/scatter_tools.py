import torch
from torch_scatter import scatter_sum, scatter_max, scatter_mean

def scatter_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min)/(f_max - f_min + eps) + index.float()*(-1)**int(descending)
    perm = norm.argsort(dim=dim, descending=descending)
    return src[perm], perm

def scatter_topk(src: torch.Tensor, index: torch.Tensor, k=1, dim=0):
  values, args = [], []
  copy = src.clone().float()
  for _ in range(k):
    v, a = scatter_max(copy, index, dim=dim)
    copy[a] = -999
    values.append(v)
    args.append(a)
  return torch.stack(values).T, torch.stack(args).T

def scatter_cumsum(src: torch.Tensor, index: torch.Tensor, dim=0, n_elements=None):
  if n_elements is None:
    _, n_elements = index.unique(return_counts=True)

  cumsum = src.cumsum(dim=dim)
  sum = scatter_sum(src, index)
  offset = sum.cumsum(dim=dim)-sum
  offset = torch.repeat_interleave(offset, n_elements)
  return cumsum - offset

def scatter_cumsum_inv(src: torch.Tensor, index: torch.Tensor, dim=0, n_elements=None):
  if n_elements is None:
    _, n_elements = index.unique(return_counts=True)

  cumsum = src.flip([dim]).cumsum(dim=dim).flip([dim])
  sum = scatter_sum(src, index)
  offset = sum.flip([dim]).cumsum(dim=dim).flip([dim])-sum
  offset = torch.repeat_interleave(offset, n_elements)
  return cumsum - offset

def scatter_randperm(src: torch.Tensor, index: torch.Tensor):
  randperm = torch.randperm( len(index) )
  index_shuffle = index[randperm]
  reindex = index_shuffle.argsort()
  randperm = randperm[reindex]
  return src[randperm], randperm
