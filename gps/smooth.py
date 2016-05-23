#!/usr/bin/env python
import numpy as np
import modest.cv
import rbf.smooth
import modest
import matplotlib.pyplot as plt
import scipy.sparse
from myplot.cm import viridis
from myplot.colorbar import pseudo_transparent_cmap
import logging
logger = logging.getLogger(__name__)

def network_smooth(u,t,x,
                   stencil_size=10,connectivity=None,order=1,
                   t_damping=None,x_damping=None,
                   basis=rbf.basis.phs3,x_vert=None,x_smp=None,
                   t_vert=None,t_smp=None,cv_itr=100,plot=False,
                   x_log_bounds=None,t_log_bounds=None,fold=10,
                   use_umfpack=True):

  if x_log_bounds is None:
    x_log_bounds = [-4.0,4.0]
  if t_log_bounds is None:
    t_log_bounds = [-4.0,4.0]
 
  u = np.asarray(u)

  Nx = x.shape[0]
  Nt = t.shape[0]

  if u.shape != (Nt,Nx):
    raise TypeError('u must have shape (Nt,Nx)')

  u_flat = u.flatten()

  # form space smoothing matrix
  Lx = rbf.smooth.smoothing_matrix(x,stencil_size=stencil_size,
                                   connectivity=connectivity,
                                   order=order,basis=basis,
                                   vert=x_vert,smp=x_smp)
  # remove any weights that are zero 
  Lx.eliminate_zeros()

  # this produces the traditional finite difference matrix for a 
  # second derivative
  Lt = rbf.smooth.smoothing_matrix(t[:,None],stencil_size=5,order='max',
                                   basis=basis,vert=t_vert,smp=t_smp)
  modest.tic('building')
  # the solution for the first timestep is defined to be zero and so 
  # we do not need the first column
  Lt = Lt[:,1:]

  Lt,Lx = rbf.smooth.grid_smoothing_matrices(Lt,Lx)

  # I will be estimating baseline displacement for each station
  # which have no regularization constraints.  
  ext = scipy.sparse.csr_matrix((Lt.shape[0],Nx))
  Lt = scipy.sparse.hstack((ext,Lt))
  Lt = Lt.tocsr()

  ext = scipy.sparse.csr_matrix((Lx.shape[0],Nx))
  Lx = scipy.sparse.hstack((ext,Lx))
  Lx = Lx.tocsr()

  # build observation matrix
  G = scipy.sparse.eye(Nx*Nt)
  G = G.tocsr()

  # chop off the first Nx columns to make room for the baseline 
  # conditions
  G = G[:,Nx:]

  # add baseline elements
  Bt = scipy.sparse.csr_matrix(np.ones((Nt,1)))
  Bx = scipy.sparse.csr_matrix((0,Nx))
  Bt,Bx = rbf.smooth.grid_smoothing_matrices(Bt,Bx)
  G = scipy.sparse.hstack((Bt,G))
  G = G.tocsc()
  #print(np.linalg.cond(G.toarray()))
  #print((scipy.sparse.linalg.inv(G),))
  #quit()
  modest.toc('building')
  # estimate damping parameters
  if (t_damping is None) & (x_damping is None):
    logger.info(
      'damping parameters were not specified and will now be '
      'estimated with cross validation')
    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=fold,plot=plot,
            log_bounds=[t_log_bounds,x_log_bounds],use_umfpack=use_umfpack)
    t_damping = out[0][0] 
    x_damping = out[0][1] 
    
  elif t_damping is None:
    logger.info(
      'time damping parameter was not specified and will now be '
      'estimated with cross validation')
    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=fold,plot=plot,
            log_bounds=[t_log_bounds,[np.log10(x_damping)-1e-4,np.log10(x_damping)+1e-4]],
            use_umfpack=use_umfpack)
    t_damping = out[0][0]

  elif x_damping is None:
    logger.info(
      'spatial damping parameter was not specified and will now be '
      'estimated with cross validation')
    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=fold,plot=plot,
            log_bounds=[[np.log10(t_damping)-1e-4,np.log10(t_damping)+1e-4],x_log_bounds],
            use_umfpack=use_umfpack)
    x_damping = out[0][1]

  L = scipy.sparse.vstack((t_damping*Lt,x_damping*Lx))
  logger.info('solving for predicted displacements ...')
  u_pred = modest.cv.sparse_direct_solve(G,L,u_flat,use_umfpack=use_umfpack)
  u_pred = u_pred.reshape((Nt,Nx))
  # zero the initial displacements
  u_pred[0,:] = 0.0
  logger.info('finished')

  return u_pred


