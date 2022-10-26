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

def scatter_jet_pair_class_reconstruction(score, batch, n=6):
    """Iteratively select the highest scoring jets with respect to class pairs
    B - batch size
    N - number of jets for each batch
    C - number of categories. *NOTE* this assumes the first category is not for signal

    Args:
        score (torch.Tensor): tensor of jet scores for each class. shape (B*N, C)
        batch (torch.Tensor): tensor of jet event indices. shape (B*N)
        n (int, optional): number of jets to select. Defaults to 6.

    Returns:
        _type_: _description_
    """
    score = score.clone() # clone the scores so we dont overwrite the original
    njets = batch.unique(return_counts=True)[1] # count the number of jets in each event

    # accumulator for the results for the selection
    values, category, jetargs = [], [], []

    for i in range(n):
        # assuming the first class is for not signal 
        # get the highest scoring category for each jet
        maxcat_value, maxcat_arg = score[:, 1:].max(dim=1)

        # select the jet with the highest scoring category 
        maxjet_value, maxjet_arg = scatter_topk(maxcat_value, batch, k=1)
        # get the category for this jet (add 1 since we removed the 0 class at the begining)
        maxjet_cat = maxcat_arg[maxjet_arg] + 1

        # set all the scores for the selected jet = -1, so it is never selected again
        score[maxjet_arg] = -1

        # handle already matched jets
        if len(category) > 0:
            # loop through already selected jets, and check if this jet is paired with it
            # since we assume only 1 jet can be paired, there should not be overlap
            paired_cats = sum([ jet_cat*(jet_cat == maxjet_cat) for jet_cat in category ])
            # repeat the paired category for the number of jets in each event
            paired_cats = paired_cats.repeat_interleave(njets).unsqueeze(1)
            # set the paired category scores for all jets = -1
            score = score.scatter(dim=1, index=paired_cats, value=-1)

        # add jet information to the accumulators
        values.append( maxjet_value )
        category.append( maxjet_cat )
        jetargs.append( maxjet_arg )

    values = torch.cat(values, dim=-1)
    category = torch.cat(category, dim=-1)
    jetargs = torch.cat(jetargs, dim=-1)

    return values, category, jetargs

