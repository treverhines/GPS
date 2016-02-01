#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.linalg
 
def is_pos_def(C):
  '''
  Returns True if C is positive definite. 
  '''
  try:
    scipy.linalg.cholesky(C)
    return True

  except scipy.linalg.LinAlgError:
    return False
  

def _update(G,d,Cd,prior,Cprior):
  '''
  Parameters
  ----------
    G: system matrix
    d: observations
    Cd: observation covariance
    prior: state prior
    Cprior: state prior covariance
  '''
  y = d - G.dot(prior)
  S = G.dot(Cprior).dot(G.T) + Cd
  K = Cprior.dot(G.T).dot(np.linalg.inv(S))
  post = prior + K.dot(y)
  Cpost = (np.eye(len(prior)) - K.dot(G)).dot(Cprior)
  return post,Cpost


def _predict(F,Q,post,Cpost):
  '''
  Parameters
  ----------
    F: state transition matrix
    Q: transition covariance matrix
    post: state posterior
    Cpost: state posterior covariance
  '''
  prior = F.dot(post)
  Cprior = F.dot(Cpost).dot(F.T) + Q
  return prior,Cprior


def _rts_smooth(post,Cpost,prior,Cprior,smooth,Csmooth,F):
  '''
  Parameters
  ----------
    post: posterior state for time i
    Cpost: posterior state covariance for time i
    prior: prior state for time i+1
    Cprior: prior state covariance for time i+1
    smooth: smooth state for time i+1
    Csmooth: smooth state covariance for time i+1
    F: transition matrix which takes the posterior from time i to the 
      prior for time i+1
  '''    
  Ck = Cpost.dot(F.T).dot(np.linalg.inv(Cprior))
  smooth_new = post + Ck.dot(smooth - prior)
  Csmooth_new = Cpost + Ck.dot(Csmooth - Cprior).dot(Ck.T)
  return smooth_new,Csmooth_new

def _stochastic_filter(u,var,t,alpha,
                       init_prior=None,
                       init_prior_cov=None,
                       diff=0):
  '''
  Models the acceleration of the observation vector as white noise with 
  variance alpha.
  '''
  u = np.asarray(u)
  # turn u into an array of data vectors
  u = u[:,None]
  var = np.asarray(var)
  # turn var into an array of covariance matrices
  var = var[:,None,None]
  t = np.asarray(t)

  N = len(t)
  # return empty arrays if the input is empty
  if N == 0:
    return np.zeros((0,2)),np.zeros((0,2))

  state_prior = np.zeros((N,2))
  state_post = np.zeros((N,2))
  state_prior_cov = np.zeros((N,2,2))
  state_post_cov = np.zeros((N,2,2))
  trans = np.zeros((N,2,2))
  system = np.array([[1.0,0.0]])

  if init_prior is None:
    init_prior = np.zeros(2)

  if init_prior_cov is None:
    init_prior_cov = 1e3*np.eye(2)

  state_prior[0,:] = init_prior
  state_prior_cov[0,:,:] = init_prior_cov

  for i in range(N):
    # incorperate observations to form posterior
    out = _update(system,u[i],var[i],state_prior[i],state_prior_cov[i])
    state_post[i] = out[0]
    state_post_cov[i] = out[1]

    # update posterior to find prior for next step    
    # Do not make a prediction for last time step
    if i != (N-1):
      dt = t[i+1] - t[i]
      trans[i+1] = np.array([[1.0,      dt],
                             [0.0,     1.0]])
      trans_cov = np.array([[0.333*dt**3, 0.500*dt**2],
                            [0.500*dt**2,          dt]])
      trans_cov *= alpha**2

      out = _predict(trans[i+1],trans_cov,state_post[i],state_post_cov[i])
      state_prior[i+1] = out[0]
      state_prior_cov[i+1] = out[1]
    
  # smooth the state variables 
  state_smooth = np.zeros((N,2))
  state_smooth_cov = np.zeros((N,2,2))
  state_smooth[-1] = state_post[-1]
  state_smooth_cov[-1] = state_post_cov[-1]

  for k in range(N-1)[::-1]:
    out = _rts_smooth(state_post[k],state_post_cov[k],
                      state_prior[k+1],state_prior_cov[k+1],
                      state_smooth[k+1],state_smooth_cov[k+1],
                      trans[k+1])
    state_smooth[k] = out[0]
    state_smooth_cov[k] = out[1]

  # check if all covariances matrices are positive definite. If not
  # then numerical errors may have significantly influenced the 
  # solution
  for i in range(N):
    if not is_pos_def(state_smooth_cov[i]):
      print('WARNING: smoothed covariance at time %s is not positive '
            'definite. This may be due to an accumulation of numerical '
            'error. Consider rescaling the input parameters or '
            'changing the initial prior and covariance' % t[i])

  # returns the indicated derivative
  out = state_smooth[:,diff]
  out_cov = state_smooth_cov[:,diff,diff]
  return out,out_cov


def stochastic_filter(u,var,t,alpha,jumps=None,init_prior=None,init_prior_cov=None,diff=None):
  ''''
  independent Kalman filters are applied to each time interval between jumps
  '''
  u = np.asarray(u)
  var = np.asarray(var)
  t = np.asarray(t)

  if jumps is None:
    jumps = np.zeros(0,dtype=float)

  jumps = np.sort(jumps)
  jumps = np.concatenate(([-np.inf],jumps,[np.inf]))
  N = len(jumps)
  out = np.zeros(len(t))
  out_cov = np.zeros(len(t))
  for i in range(N-1):
    idx = (t > jumps[i]) & (t <= jumps[i+1])  
    a,b = _stochastic_filter(u[idx],var[idx],t[idx],alpha,
                             init_prior=init_prior,
                             init_prior_cov=init_prior_cov,
                             diff=diff)
    out[idx] = a
    out_cov[idx] = b

  return out,out_cov


def running_mean(u,var,t,Ns=10,jumps=None):
  if jumps is None:
    jumps = np.zeros(0,dtype=float)

  vert = np.array(jumps,dtype=float)[:,None]
  smp = np.arange(vert.shape[0],dtype=int)[:,None]
  s,dx = rbf.stencil.nearest(t[:,None],t[:,None],Ns,vert,smp)
  u_mean = np.array([np.average(u[i],weights=1.0/var[i]) for i in s])
  var_mean = np.array([1.0/np.sum(1.0/var[i]) for i in s])
  return u_mean,var_mean
