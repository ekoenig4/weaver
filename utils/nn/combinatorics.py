import torch, itertools
import numpy as np

# v2 is faster on CPU maybe...
@torch.jit.script
def get_group_feature_v2(x, group_idx):
    """Group features in x with indicies in group_idx
    B - batch dimension
    F - feature dimension
    N - object dimension
    I - number of objects in each group
    G - number of groups 
    Args:
        x (Tensor): Feature tensor to group, shape (B x F x N)
        group_idx (LongTensor): Index groups to group feature array, shape (B x I x G)

    Returns:
        Tensor: New group feature tensor, shape (B x F x I x G)
    """
    batch_size, num_dims, num_points = x.size()
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
    group_idx = group_idx + idx_base

    fts = x.transpose(0,1).reshape(num_dims, -1)
    fts = fts[:, group_idx]
    fts = fts.transpose(1,0).contiguous()
    return fts


# v1 is faster on GPU maybe...
@torch.jit.script
def get_group_feature_v1(x, group_idx):
    """Group features in x with indicies in group_idx
    B - batch dimension
    F - feature dimension
    N - object dimension
    I - number of objects in each group
    G - number of groups 
    Args:
        x (Tensor): Feature tensor to group, shape (B x F x N)
        group_idx (LongTensor): Index groups to group feature array, shape (B x I x G)

    Returns:
        Tensor: New group feature tensor, shape (B x F x I x G)
    """

    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    group_idx = group_idx.to(x.device) + idx_base

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[group_idx, :]
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_groups, num_points)
    return fts

def p4_to_cartesian(m, pt, eta, phi):
  """Converts collider p4 to cartesian p4

  Args:
      m (Tensor): Tensor of mass
      pt (Tensor): Tensor of pt
      eta (Tensor): Tensor of eta
      phi (Tensor): Tensor of phi

  Returns:
      tuple: Tuple of cartesian p4 (E, Px, Py, Pz)
  """
  px = pt*torch.cos(phi)
  py = pt*torch.sin(phi)
  pz = pt*torch.sinh(eta)
  p = pt*torch.cosh(eta)
  e = torch.sqrt((m)**2 + (p)**2)
  return e, px, py, pz

def array_to_cartesian(array):
  """Converts collider array to cartesian array

  Args:
      array (Tensor): Tensor of shape (batches, features, objects). Where features are ordered pt, m, eta, phi

  Returns:
      Tensor: Tensor of cartesian p4
  """
  e, px, py, pz = p4_to_cartesian(array[:,1], array[:,0], array[:,2], array[:,3])
  return torch.stack([e, px, py, pz], dim=1)

def p4_to_collider(e, px, py, pz):
  """Converts cartesian p4 to collider p4

  Args:
      e (Tensor): Tensor of energy
      px (Tensor): Tensor of Px
      py (Tensor): Tensor of Py
      pz (Tensor): Tensor of Pz

  Returns:
      tuple: Tuple of collider p4 (M, Pt, Eta, Phi)
  """
  pt = torch.sqrt( px**2 + py**2 )
  phi = torch.atan(py/px)
  eta = torch.asinh(pz/pt)
  m2 = e**2 - (pt*torch.cosh(eta))**2
  m = torch.where(m2 < 0, -torch.sqrt( -m2 ), torch.sqrt(m2)) 
  return m, pt, eta, phi

def array_to_collider(array):
  """Converts cartesian array to collider array

  Args:
      array (Tensor): Tensor of shape (batches, features, objects). Where features are ordered e, px, py, pz

  Returns:
      Tensor: Tensor of collider p4
  """
  m, pt, eta, phi = p4_to_collider(array[:,0], array[:,1], array[:,2], array[:,3])
  return torch.stack([pt, m, eta, phi], dim=1)

def _calc_p4(m, pt, eta, phi, params=None):
  """Calculate p4 vector for grouped objects. Feature vector shape (batch x features x p4 x groups). 
  Features should be order pt, m, eta, phi

  Args:
      group_x (Tensor): Feature vector containing p4 of objects to be added together
  """
  
  if params is not None:
    unscale = lambda x, param: x/param['scale']+param['center']
    m = unscale(m, params['m'])
    pt = unscale(pt, params['pt'])
    eta = unscale(eta, params['eta'])
    phi = unscale(phi, params['phi'])

  e, px, py, pz = p4_to_cartesian(m, pt, eta, phi)

  px = px.sum(dim=1)
  py = py.sum(dim=1)
  pz = pz.sum(dim=1)
  e = e.sum(dim=1)

  m, pt, eta, phi = p4_to_collider(e, px, py, pz)

  if params is not None:
    scale = lambda x, param: param['scale']*(x-param['center'])
    m = scale(m, params['m'])
    pt = scale(pt, params['pt'])
    eta = scale(eta, params['eta'])
    phi = scale(phi, params['phi'])

  p4 = torch.stack([pt, m, eta, phi], dim=1)
  return p4

def calc_group_p4(group_x, params=None):
  """Calculate p4 vector for grouped objects. Feature vector shape (batch x features x p4 x groups). 
  Features should be order pt, m, eta, phi

  Args:
      group_x (Tensor): Feature vector containing p4 of objects to be added together
  """
  return _calc_p4(group_x[:,1], group_x[:,0], group_x[:,2], group_x[:,3], params=params)

def _combinations(items, ks):
   if len(ks) == 1:
      for c in itertools.combinations(items, ks[0]):
         yield (c,)

   else:
      for c_first in itertools.combinations(items, ks[0]):
         items_remaining= set(items) - set(c_first)
         for c_other in \
           _combinations(items_remaining, ks[1:]):
            if len(c_first)!=len(c_other[0]) or c_first<c_other[0]:
               yield (c_first,) + c_other    

def torch_combinations(nitems, ks):
   combs = list(_combinations(np.arange(nitems), ks))
   return torch.Tensor(combs)