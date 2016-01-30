#!/usr/bin/env python
from __future__ import division
import numpy as np
 
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


def _kalman_filter(u,ucov,t,alpha):
  '''
  continuous Kalman filter
  '''
  u = np.asarray(u)
  ucov = np.asarray(ucov)
  t = np.asarray(t)

  N = len(t)
  state_prior = np.zeros((N,3))
  state_post = np.zeros((N,3))
  state_prior_cov = np.zeros((N,3,3))
  state_post_cov = np.zeros((N,3,3))
  trans = np.zeros((N,3,3))
  system = np.array([[1.0,0.0,0.0]])

  # assume no prior
  state_prior_cov[0,:,:] = 1e6*np.eye(3)

  for i in range(N):
    # incorperate observations to form posterior
    out = _update(system,u[[i]],ucov[[i]],state_prior[i],state_prior_cov[i])
    state_post[i] = out[0]
    state_post_cov[i] = out[1]

    # update posterior to find prior for next step    
    # Do not make a prediction for last time step
    if i != (N-1):
      dt = t[i+1] - t[i]
      trans[i+1] = np.array([[1.0,      dt, 0.5*dt**2],
                             [0.0,     1.0,        dt],
                             [0.0,     0.0,       1.0]])
      trans_cov = np.array([[0.050*dt**5, 0.125*dt**4, 0.166*dt**3],
                            [0.125*dt**4, 0.333*dt**3, 0.500*dt**2],
                            [0.166*dt**3, 0.500*dt**2,          dt]])
      trans_cov *= alpha**2

      out = _predict(trans[i+1],trans_cov,state_post[i],state_post_cov[i])
      state_prior[i+1] = out[0]
      state_prior_cov[i+1] = out[1]
    
  # smooth the state variables 
  state_smooth = np.zeros((N,3))
  state_smooth_cov = np.zeros((N,3,3))
  state_smooth[N-1] = state_post[N-1]
  state_smooth_cov[N-1] = state_post_cov[N-1]

  for k in range(N-1)[::-1]:
    out = _rts_smooth(state_post[k],state_post_cov[k],
                     state_prior[k+1],state_prior_cov[k+1],
                     state_smooth[k+1],state_smooth_cov[k+1],
                     trans[k+1])
    state_smooth[k] = out[0]
    state_smooth_cov[k] = out[1]

  state_smooth_cov = state_smooth_cov[:,[0,1,2],[0,1,2]]
  return state_smooth,state_smooth_cov


def kalman_filter(u,ucov,t,alpha,jumps):
  ''''
  independent Kalman filters are applied to each time interval between jumps
  '''
  state_smooth = np.zeros((0,3))
  state_smooth_cov = np.zeros((0,3))
  jumps = np.sort(jumps)
  jumps = np.concatenate(([-np.inf],jumps,[np.inf]))
  N = len(jumps)
  for i in range(N-1):
    idx = (t > jumps[i]) & (t <= jumps[i+1])  
    a,b = _kalman_filter(u[idx],ucov[idx],t[idx],alpha)
    state_smooth = np.vstack((state_smooth,a))
    state_smooth_cov = np.vstack((state_smooth_cov,b))

  return state_smooth,state_smooth_cov

