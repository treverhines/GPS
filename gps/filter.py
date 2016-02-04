#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.linalg
import modest
import matplotlib.pyplot as plt
 
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
  state_smooth = np.zeros((N,M))
  state_smooth_cov = np.zeros((N,M,M))
  trans = np.zeros((N,M,M))

  # set initial prior covariance
  state_prior_cov[0,:,:] = init_prior_var*np.eye(M)

  for i in range(N):
    system = np.zeros((1,M))
    system[0,0] = 1.0
    for j,val in enumerate(jumps):
      system[0,2+j] = _H(t[i]-val)

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


def _H(t):
  '''
  heaviside function
  '''
  return (t>=0.0).astype(float)


def _pslog(t):
  '''
  returns:
    log(1 + t) if t >= 0 
    0 if t < 0
  '''
  return np.log(1 + _H(t)*t)


def log_filter(u,var,t,start,jumps,trials=10,diff=0):
  '''
  Assumes that the underlying signal can be described by
  
    u(t) = a + b*t + 
           c*H(t-t_eq) + 
           d*H(t-t_eq)*(t-t_eq) +  
           e_i*log(1 + t/tau_i)
  
  and that the observation contains seasonal signals and jumps at 
  indicated times. So the observation equation is
  
    uobs(t) = a + b*t + 
              c*H(t-t_eq) + 
              d_i*log(1 + t/tau_i) + 
              f*sin(2*pi*t) + g*sin(4*pi*t) 
              h*cos(2*pi*t) + m*cos(4*pi*t) 
              n_i*H(t-t_i)

  This function estimates a,b,c,d_i,tau_i,e,f,g,h, and m_i with 
  a nonlinear least squares algorithm.  We put a positivity constraint
  on tau and we also restrict c and d_i to have the same sign
  '''
  J = len(jumps)
  # total number of model parameters 
  M = 14 + J 
  def system(m,diff=0):
    if diff == 0:
      out = np.zeros(len(t)) 
      out += m[0]  
      out += m[1]*t
      out += m[2]*_H(t-start)
      out += m[3]*_H(t-start)*(t-start)
      out += m[4]*_pslog((t-start)/m[5])
      out += m[6]*_pslog((t-start)/m[7])
      out += m[8]*_pslog((t-start)/m[9])
      out += m[10]*np.sin(2*np.pi*t)  
      out += m[11]*np.sin(4*np.pi*t)  
      out += m[12]*np.cos(2*np.pi*t)  
      out += m[13]*np.cos(4*np.pi*t)  
      for j,val in enumerate(jumps):
        out += m[14+j]*_H(t-val)  

    # derivative w.r.t. t
    elif diff == 1:
      out = np.zeros(len(t)) 
      out += 0.0  
      out += m[1]
      out += 0.0
      out += m[3]*_H(t-start)
      out += _H(t-start)*m[4]/((t-start) + m[5])
      out += _H(t-start)*m[6]/((t-start) + m[7])
      out += _H(t-start)*m[8]/((t-start) + m[9])
      out += m[10]*2*np.pi*np.cos(2*np.pi*t)  
      out += m[11]*4*np.pi*np.cos(4*np.pi*t)  
      out += -m[12]*2*np.pi*np.sin(2*np.pi*t)  
      out += -m[13]*4*np.pi*np.sin(4*np.pi*t)  

    return out
  
  nuisance_indices = np.arange(10,M,dtype=int)
  signal_indices = np.arange(10,dtype=int)
  jacobian = modest.make_jacobian(system)

  best_err = np.inf
  for i in range(trials):
    # set initial model parameter guess
    minit = np.ones(M)
    minit[5] = 10**np.random.normal(-1,0.5)
    minit[7] = 10**np.random.normal(0,0.5)
    minit[9] = 10**np.random.normal(1,0.5)
    
    # set lower and upper bounds for model parameters
    lb = -1e10*np.ones(M)
    ub = 1e10*np.ones(M)
    # do not allow timescales less that 0.001
    lb[[5,7,9]] = 1e-3
    # enforce postseismic signal to be positive
    lb[[4,6,8]] = 0.0
    err1,m1,mcov1,pred1 = modest.nonlin_lstsq(
                            system,u,minit,
                            solver=modest.bvls,solver_args=(lb,ub),
                            data_covariance=var,
                            output=['misfit','solution','solution_covariance','predicted'])
    if err1 < best_err:
      best_err = err1
      m = m1
      mcov = mcov1
      pred = pred1

    lb = -1e10*np.ones(M)
    ub = 1e10*np.ones(M)
    lb[[5,7,9]] = 1e-3
    # enforce postseismic signal to be negative
    ub[[4,6,8]] = 0.0
    err2,m2,mcov2,pred2 = modest.nonlin_lstsq(
                            system,u,minit,
                            solver=modest.bvls,solver_args=(lb,ub),
                            data_covariance=var,
                            output=['misfit','solution','solution_covariance','predicted'])
    if err2 < best_err:
      best_err = err2 
      m = m2
      mcov = mcov2
      pred = pred2

  # find the jacobian matrix for the final model estimate
  jac = jacobian(m,diff=diff)  

  # find the estimated signal by evaluating the system with 0.0 for
  # the nuisance parameters
  m[nuisance_indices] = 0.0
  signal_pred = system(m,diff=diff)

  # slice the model covariance matrix and jacobian so that it only
  # contains columns for the parameters of interest
  jac = jac[:,signal_indices]
  mcov = mcov[np.ix_(signal_indices,signal_indices)]

  # use error propagation to find the uncertainty on the predicted
  # signal
  signal_cov = jac.dot(mcov).dot(jac.T)
  signal_var = np.diag(signal_cov) 

  return signal_pred,signal_var  


def stochastic_log_filter(u,var,t,alpha,teq,jumps=None,init_prior_var=1e3,diff=0):
  '''
  let u(t) be the true signal we are trying to recover and obs(t) be the observation
  of that signal which is obscured by jumps and seasonal processes. we say that our
  true signal is a sum of logarithms and integrated Brownian motion, B(t)

    u(t) = a + b*t + 
           H(t-t_eq)*(c*log(1 + (t-t_eq)/0.1) + 
                      d*log(1 + (t-t_eq)/1.0) +  
                      e*log(1 + (t-t_eq)/10.0) + B(t))

  our observation is 

    obs(t) = u(t) + d*sin(2*pi*t) + e*sin(4*pi*t) +  
                    f*cos(2*pi*t) + g*cos(4*pi*t) + 
                    h_i*H(t-t_i)
  '''
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
  # the state variable consists of 2 parameters for the integrated
  # brownian motion, 3 parameters for the logarithms, 4 seasonal
  # parameters, a secular velocity and baseline displacement, 
  # plus however many jumps were given, 
  M = 11 + J

  # return empty arrays if the time length is zero
  if N == 0:
    return np.zeros(0),np.zeros(0)

  # remove jumps that are not within the observation time interval
  jumps = np.array([j for j in jumps if (j>t[0]) & (j<t[-1])])

  state_prior = np.zeros((N,M))
  state_post = np.zeros((N,M))
  state_prior_cov = np.zeros((N,M,M))
  state_post_cov = np.zeros((N,M,M))
  state_smooth = np.zeros((N,M))
  state_smooth_cov = np.zeros((N,M,M))
  trans = np.zeros((N,M,M))
  trans_cov = np.zeros((N,M,M))
  system = np.zeros((N,M))

  signal_indices = np.arange(7,dtype=int)
  nuisance_indices = np.arange(7,M,dtype=int)

  # set initial prior covariance
  state_prior_cov[0,:,:] = init_prior_var*np.eye(M) 

  # build system matrix for every time step
  system[:,0] = _H(t-teq)
  # Brownian motion
  system[:,1] = 0.0
  # baseline
  system[:,2] = 1.0
  # secular velocity
  system[:,3] = t
  # first log term
  system[:,4] = _pslog((t-teq)/0.1)
  # second log term
  system[:,5] = _pslog((t-teq)/1.0)
  # third log term
  system[:,6] = _pslog((t-teq)/10.0)
  # first annual seasonal term
  system[:,7] = np.sin(2*np.pi*t)
  # second annual seasonal term
  system[:,8] = np.cos(2*np.pi*t)
  # first semi-annual seasonal term
  system[:,9] = np.sin(4*np.pi*t)
  # second semi-annual seasonal term
  system[:,10] = np.cos(4*np.pi*t)
  for j,val in enumerate(jumps):
    system[:,11+j] = _H(t-val)

  # build transition matrix for every time step
  dt = np.diff(t)
  trans[1:] = np.eye(M)
  trans[1:,0,1] = dt
  trans_cov[1:,0,0] = alpha**2*0.333*dt**3 
  trans_cov[1:,0,1] = alpha**2*0.500*dt**2 
  trans_cov[1:,1,0] = alpha**2*0.500*dt**2 
  trans_cov[1:,1,1] = alpha**2*dt
  
  for i in range(N):
    # use observations to form posterior
    out = _update(system[[i]],u[i],var[i],state_prior[i],state_prior_cov[i])
    state_post[i] = out[0]
    state_post_cov[i] = out[1]
    # update posterior to find prior for next step    
    # Do not make a prediction for last time step
    if i != (N-1):
      #dt = t[i+1] - t[i]
      #trans[i+1] = np.eye(M)
      #trans[i+1,0,1] = dt
      #trans_cov = np.zeros((M,M))
      #trans_cov[0,0] = 0.333*dt**3 
      #trans_cov[0,1] = 0.500*dt**2 
      #trans_cov[1,0] = 0.500*dt**2 
      #trans_cov[1,1] = dt
      #trans_cov *= alpha**2
      out = _predict(trans[i+1],trans_cov[i+1],state_post[i],state_post_cov[i])
      state_prior[i+1] = out[0]
      state_prior_cov[i+1] = out[1]
    
  # smooth the state variables 
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
  #out = state_smooth[:,diff]
  #out_var = state_smooth_cov[:,diff,diff]
  # compute predicted data

  state_smooth[:,nuisance_indices] = 0.0
  out = np.einsum('...i,...i',system,state_smooth)
  out_var = np.ones(N)
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
