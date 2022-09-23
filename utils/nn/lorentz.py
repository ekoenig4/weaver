import functools
import vector
import torch

def torch_p4(array: torch.Tensor, keys=['pt','m','eta','phi']) -> vector.MomentumNumpy4D:
  return vector.obj(**{key:array[:,i].cpu().numpy() for i, key in enumerate(keys)})

def p4_torch(p4: vector.MomentumNumpy4D, keys=['pt','m','eta','phi']) -> torch.Tensor:
  return torch.stack([ torch.from_numpy( getattr(p4,key) ) for key in keys ], dim=1)

def sum_p4(array, keys=['pt','m','eta','phi']):
  dev = array.device
  batch, features, objects = array.shape
  obj_p4 = [ torch_p4(array[:,:,i], keys=keys) for i in range(objects) ]
  total_p4 = functools.reduce(vector.Lorentz.add, obj_p4)
  return p4_torch(total_p4).unsqueeze(2).to(dev)

def boost_array(array, boost, keys=['pt','m','eta','phi']):
  if len(array.shape) < 3: array = array.unsqueeze(2)
  if len(boost.shape) < 3: boost = boost.unsqueeze(2)

  array_p4 = torch_p4(array, keys=keys)
  boost_p4 = torch_p4(boost, keys=keys)
  boosted_p4 = array_p4.boost_p4(-boost_p4)
  return p4_torch(boosted_p4)