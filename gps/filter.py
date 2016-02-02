#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.linalg
 
def heaviside(t):
  t = np.asarray(t)
  return (t>=0.0).astype(float)


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
                       init_prior_var=1e3,
                       jumps=None,
                       diff=0):
  u = np.asarray(u)
  # turn u into an array of data vectors
  u = u[:,None]
  var = np.asarray(var)
  # turn var into an array of covariance matrices
  var = var[:,None,None]
  t = np.asarray(t)

  if jumps is None:
    jumps = []

  jumps = np.asarray(jumps)

  N = t.shape[0]
  J = jumps.shape[0]
  M = 2 + J

  # return empty arrays if the time length is zero
  if N == 0:
    return np.zeros(0),np.zeros(0)

  # remove jumps that are not within the observation time interval
  jumps = np.array([j for j in jumps if (j>t[0]) & (j<t[-1])])

  state_prior = np.zeros((N,M))
  state_post = np.zeros((N,M))
  state_prior_cov = np.zeros((N,M,M))
  state_post_cov = np.zeros((N,M,M))
  trans = np.zeros((N,M,M))

  # set initial prior covariance
  state_prior_cov[0,:,:] = init_prior_var*np.eye(M)

  for i in range(N):
    system = np.zeros((1,M))
    system[0,0] = 1.0
    for j,val in enumerate(jumps):
      system[0,2+j] = heaviside(t[i]-val)

    # use observations to form posterior
    out = _update(system,u[i],var[i],state_prior[i],state_prior_cov[i])
    state_post[i] = out[0]
    state_post_cov[i] = out[1]

    # update posterior to find prior for next step    
    # Do not make a prediction for last time step
    if i != (N-1):
      dt = t[i+1] - t[i]
      trans[i+1] = np.eye(M)
      trans[i+1,0,1] = dt
      trans_cov = np.zeros((M,M))
      trans_cov[0,0] = 0.333*dt**3 
      trans_cov[0,1] = 0.500*dt**2 
      trans_cov[1,0] = 0.500*dt**2 
      trans_cov[1,1] = dt
      trans_cov *= alpha**2

      out = _predict(trans[i+1],trans_cov,state_post[i],state_post_cov[i])
      state_prior[i+1] = out[0]
      state_prior_cov[i+1] = out[1]
    
  # smooth the state variables 
  state_smooth = np.zeros((N,M))
  state_smooth_cov = np.zeros((N,M,M))
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
  out_var = state_smooth_cov[:,diff,diff]
  return out,out_var


def stochastic_filter(u,var,t,alpha,
                      restarts=None,
                      jumps=None,
                      init_prior_var=1e3,
                      diff=0):
  '''
  Description
  -----------
    Uses a Kalman filter to estimate a smooth underlying signal from
    noisy, and possibly discontinous, data.

    The acceleration of the underlying signal is modeled as white
    noise with standard deviation alpha. A Kalman filter is used to
    estimate the underlying signal and also estimate magnitudes of
    jumps at known times.
  
    The observation function is
   
      u_obs(t) = u_true(t) + a_0*H(t-t_0) + ... a_J*H(t-t_J) + sigma
  
    
    where sigma is is the observation noise, H(t) is the Heaviside
    function, t_i are the jump times, a_i are the jump magnitudes that
    will be estimates. u_true(t) is the true signal we are trying to
    estimate, which is modeled as a stochastic process.

    Our state vector consists of the true signal, its time 
    derivative and the jump coefficients:

      X(t) = [u_true(t), u_true'(t), a_0, ... a_J]. 

    The state transition function is 
   
      X(t+1) = F(dt)*X(t) + epsilon

    where epsilon is the process noise and 

      F(dt) = [1.0  dt 0.0     0.0]
              [0.0 1.0 0.0 ... 0.0] 
              [0.0 0.0 1.0     0.0]
              [     :          :  ]
              [0.0 0.0 0.0 ... 1.0].

    epsion has zero mean and covariance described by

      Q(dt) = [dt**3/3 dt**2/2 0.0 ... 0.0]   
              [dt**2/2      dt 0.0 ... 0.0]
              [0.0         0.0 0.0     0.0]       
              [     :                  :  ]
              [0.0         0.0 0.0 ... 0.0].

  Returns
  -------
    the estimate of u_true and its variance or the time derivative 
    of u_true and its variance. 

  '''
  u = np.asarray(u)
  var = np.asarray(var)
  t = np.asarray(t)

  if restarts is None:
    restarts = np.zeros(0,dtype=float)

  restarts = np.sort(restarts)
  restarts = np.concatenate(([-np.inf],restarts,[np.inf]))
  out = np.zeros(t.shape[0])
  out_var = np.zeros(t.shape[0])
  for i in range(restarts.shape[0]-1):
    idx = (t >= restarts[i]) & (t < restarts[i+1])  
    a,b = _stochastic_filter(u[idx],var[idx],t[idx],alpha,
                             jumps=jumps,
                             init_prior_var=init_prior_var,
                             diff=diff)
    out[idx] = a
    out_var[idx] = b

  return out,out_var


def running_mean(u,var,t,Ns=10,jumps=None):
  if jumps is None:
    jumps = np.zeros(0,dtype=float)

  vert = np.array(jumps,dtype=float)[:,None]
  smp = np.arange(vert.shape[0],dtype=int)[:,None]
  s,dx = rbf.stencil.nearest(t[:,None],t[:,None],Ns,vert,smp)
  u_mean = np.array([np.average(u[i],weights=1.0/var[i]) for i in s])
  var_mean = np.array([1.0/np.sum(1.0/var[i]) for i in s])
  return u_mean,var_mean
