#!/usr/bin/env python
import numpy as np

class MaskedNearestInterp:
  def __init__(self,x,y,tol=np.inf):
    self.x = x
    self.y = y
    self.tol = tol

  def __call__(self,x):
    isscalar = np.isscalar(x)
    nearest_idx = np.argmin(np.abs(self.x[:,None] - x),0) 
    if isscalar:
      nearest_idx = nearest_idx[0]

    data = self.y[nearest_idx]
    mask = self.invalid_mask(x)
    return np.ma.masked_array(data,mask=mask)

  def invalid_mask(self,x):
    isscalar = np.isscalar(x)
    nearest = np.min(np.abs(self.x[:,None] - x),0)
    if isscalar:
      nearest = nearest[0]

    x_mask = np.asarray(nearest > self.tol,dtype=int)
    itp_shape = list(np.shape(self.y))
    itp_shape.pop(0)
    if isscalar:
      itp_mask = np.zeros(itp_shape,dtype=int)
      itp_mask += x_mask
      return itp_mask

    else:
      itp_shape.insert(0,len(x))
      itp_mask = np.zeros(itp_shape,dtype=int)
      # reshape the interpolation mask so that the x mask can       
      # be broadcasted on it                     
      #itp_mask = swap_axes(itp_mask,self.axis,-1)    
      itp_mask = np.swapaxes(itp_mask,0,-1)
      itp_mask += x_mask
      #itp_mask = swap_axes(itp_mask,self.axis,-1)  
      itp_mask = np.swapaxes(itp_mask,0,-1)
      itp_mask = np.array(itp_mask,dtype=bool) 
      return itp_mask



    
