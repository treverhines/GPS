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

def _psexp(t):
  ''' 
  returns:
    1 - exp(-t) if t >= 0 
    0 if t < 0
  '''
  return 1-np.exp(-_H(t)*t)


def logexp_filter(u,var,t,start,jumps,trials=10,diff=0,detrend=False,reg=1e-10):
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
  # remove jumps that are not within the observation time interval
  jumps = np.array([j for j in jumps if (j>t[0]) & (j<t[-1])])

  J = len(jumps)
  # total number of model parameters 
  M = 11 + J 

  def system(m,diff=0):
    if diff == 0:
      out = np.zeros(len(t)) 
      out += m[0]  
      out += m[1]*t
      out += m[2]*_H(t-start)
      out += m[3]*_pslog((t-start)/m[4])
      out += m[5]*_psexp((t-start)/m[6])
      out += m[7]*np.sin(2*np.pi*t)  
      out += m[8]*np.sin(4*np.pi*t)  
      out += m[9]*np.cos(2*np.pi*t)  
      out += m[10]*np.cos(4*np.pi*t)  
      for j,val in enumerate(jumps):
        out += m[11+j]*_H(t-val)  

    # derivative w.r.t. t
    elif diff == 1:
      out = np.zeros(len(t)) 
      out += m[1]
      out += _H(t-start)*(m[3]/((t-start) + m[4]))
      out += _H(t-start)*(m[5]*np.exp(-(t-start)/m[6])/m[6])
      out += m[7]*np.cos(2*np.pi*t)*2*np.pi
      out += m[8]*np.cos(4*np.pi*t)*4*np.pi
      out += -m[9]*np.sin(2*np.pi*t)*2*np.pi  
      out += -m[10]*np.sin(4*np.pi*t)*4*np.pi

    return out
  
  if detrend:  
    idx1 = np.array([0,1,7,8,9,10])
    idx2 = np.arange(11,M,dtype=int)
    nuisance_indices = np.concatenate((idx1,idx2))
    signal_indices = np.array([2,3,4,5,6])

  else:
    idx1 = np.array([7,8,9,10])
    idx2 = np.arange(11,M,dtype=int)
    nuisance_indices = np.concatenate((idx1,idx2))
    signal_indices = np.array([0,1,2,3,4,5,6])

  jacobian = modest.make_jacobian(system)

  best_err = np.inf
  for i in range(trials):
    # set lower and upper bounds for model parameters
    minit = np.ones(M)
    lb = -1e10*np.ones(M)
    ub = 1e10*np.ones(M)
    # do not allow timescales less that 0.001
    minit[4] = 10**np.random.normal(0.0,0.5)
    minit[6] = 10**np.random.normal(0.0,0.5)
    lb[4] = 1e-3        
    lb[6] = 1e-3        

    err1,m1,mcov1,pred1 = modest.nonlin_lstsq(
                            system,u,minit,
                            solver=modest.bvls,solver_args=(lb,ub),
                            data_covariance=var,
                            LM_damping=True, 
                            regularization=(0,reg),
                            output=['misfit','solution','solution_covariance','predicted'])
    if err1 < best_err:
      best_err = err1
      m = m1
      mcov = mcov1
      pred = pred1

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
  # for some reason, diag returns a read-only array
  signal_var = np.copy(np.diag(signal_cov))

  return signal_pred,signal_var  



def log_filter(u,var,t,start,jumps,logs=3,trials=10,diff=0,detrend=False,reg=1e-10):
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
  # remove jumps that are not within the observation time interval
  jumps = np.array([j for j in jumps if (j>t[0]) & (j<t[-1])])

  # determine if there is any preseismic data
  has_preseismic = np.any(t<start)

  L = logs
  #logscales = 10.0**np.arange(1,1-L,-1)
  logscales = 10.0**np.linspace(1,-1,L)
  J = len(jumps)
  # total number of model parameters 
  M = 7 + J + 2*L

  def system(m,diff=0):
    if diff == 0:
      out = np.zeros(len(t)) 
      out += m[0]  
      out += m[1]*t
      out += m[2]*_H(t-start)
      for i in range(L):
        out += m[3+2*i]*np.log(1 + _H(t-start)*(t-start)/m[3+2*i+1])

      # next index is 2 + 2*L
      out += m[3+2*L]*np.sin(2*np.pi*t)  
      out += m[4+2*L]*np.sin(4*np.pi*t)  
      out += m[5+2*L]*np.cos(2*np.pi*t)  
      out += m[6+2*L]*np.cos(4*np.pi*t)  
      for j,val in enumerate(jumps):
        out += m[7+2*L+j]*_H(t-val)  

    # derivative w.r.t. t
    elif diff == 1:
      out = np.zeros(len(t)) 
      out += 0.0  
      out += m[1]

      for i in range(L):      
        out += m[3+2*i]*_H(t-start)/((t-start) + m[3+2*i+1])

      out += m[3+2*L]*2*np.pi*np.cos(2*np.pi*t)  
      out += m[4+2*L]*4*np.pi*np.cos(4*np.pi*t)  
      out += -m[5+2*L]*2*np.pi*np.sin(2*np.pi*t)  
      out += -m[6+2*L]*4*np.pi*np.sin(4*np.pi*t)  

    return out
  
  if detrend:  
    idx1 = np.array([0,1])
    idx2 = np.arange(3+2*L,M,dtype=int)
    nuisance_indices = np.concatenate((idx1,idx2))
    signal_indices = np.arange(2,3+2*L,dtype=int)
  else:
    nuisance_indices = np.arange(3+2*L,M,dtype=int)
    signal_indices = np.arange(3+2*L,dtype=int)

  jacobian = modest.make_jacobian(system)

  best_err = np.inf
  for i in range(trials):
    # set lower and upper bounds for model parameters
    minit = np.ones(M)
    lb = -1e10*np.ones(M)
    ub = 1e10*np.ones(M)
    # do not allow timescales less that 0.001
    for i,val in enumerate(logscales):
      minit[3+2*i+1] = 10**np.random.normal(np.log10(val),0.25)
      lb[3+2*i+1] = 1e-3        

    #lb[[4,6,8]] = 0.0
    err1,m1,mcov1,pred1 = modest.nonlin_lstsq(
                            system,u,minit,
                            solver=modest.bvls,solver_args=(lb,ub),
                            data_covariance=var,
                            LM_damping=True, 
                            regularization=(0,reg),
                            output=['misfit','solution','solution_covariance','predicted'])
    if err1 < best_err:
      best_err = err1
      m = m1
      mcov = mcov1
      pred = pred1

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
  # for some reason, diag returns a read-only array
  signal_var = np.copy(np.diag(signal_cov))

  return signal_pred,signal_var  


def stochastic_filter(u,var,t,alpha=0.1,signal_jumps=None,noise_jumps=None,
                      init_prior=0.0,init_prior_var=1.0,diff=0):
  u = np.asarray(u)
  # turn u into an array of data vectors
  u = u[:,None]
  var = np.asarray(var)
  # turn var into an array of covariance matrices
  var = var[:,None,None]

  t = np.asarray(t)
  
  if signal_jumps is None:
    signal_jumps = []

  if noise_jumps is None:
    noise_jumps = []

  signal_jumps = np.asarray(signal_jumps)
  noise_jumps = np.asarray(noise_jumps)

  # remove jumps that are not within the observation time interval
  signal_jumps = np.array([j for j in signal_jumps if (j>t[0]) & (j<t[-1])])
  noise_jumps = np.array([j for j in noise_jumps if (j>t[0]) & (j<t[-1])])
  jumps = np.concatenate((signal_jumps,noise_jumps))
  N = t.shape[0]
  JN = noise_jumps.shape[0]
  JS = signal_jumps.shape[0]
  J = jumps.shape[0] 
  # the state variable consists of 2 parameters for u and the rate of u 
  # as well as a parameter for each jump and 4 seasonal terms
  M = 6 + J 

  # return empty arrays if there are no observations
  if N == 0:
    return np.zeros(0),np.zeros(0)

  state_prior = np.zeros((N,M))
  state_post = np.zeros((N,M))
  state_prior_cov = np.zeros((N,M,M))
  state_post_cov = np.zeros((N,M,M))
  state_smooth = np.zeros((N,M))
  state_smooth_cov = np.zeros((N,M,M))
  trans = np.zeros((N,M,M))
  trans_cov = np.zeros((N,M,M))
  system = np.zeros((N,M))

  # set initial prior covariance
  state_prior_cov[0,:,:] = init_prior_var*np.eye(M) 
  state_prior[0,:] = init_prior*np.ones(M)

  # build system matrix for every time step
  # Brownian motion
  system[:,0] = 1.0
  # Brownian motion velocity
  system[:,1] = 0.0
  for j,val in enumerate(jumps):
    system[:,2+j] = _H(t-val)

  # first annual seasonal term
  system[:,2+J] = np.sin(2*np.pi*t)
  # second annual seasonal term
  system[:,2+J+1] = np.cos(2*np.pi*t)
  # first semi-annual seasonal term
  system[:,2+J+2] = np.sin(4*np.pi*t)
  # second semi-annual seasonal term
  system[:,2+J+3] = np.cos(4*np.pi*t)


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

  # return the prediction to the data without the noise jumps
  if diff == 0:
    state_smooth = state_smooth[:,:-(JN+4)]
    state_smooth_cov = state_smooth_cov[:,:-(JN+4),:]
    state_smooth_cov = state_smooth_cov[:,:,:-(JN+4)]
    system = system[:,:-(JN+4)]
    out = np.einsum('...i,...i',system,state_smooth)
    out_var = np.einsum('...i,...ij,...j',system,state_smooth_cov,system)
  elif diff == 1:
    out = state_smooth[:,1]
    out_var = state_smooth_cov[:,1,1]

  return out,out_var


def stochastic_detrender(u,var,t,teq=0.0,alpha=0.1,jumps=None,prior_vel=None,prior_vel_var=None,
                         init_prior_var=1.0,diff=0,detrend=False):
  ''' 
  let u(t) be the true signal we are trying to recover and obs(t) be the observation
  of that signal which is obscured by jumps and seasonal processes. we say that our
  true signal is a sum of logarithms and integrated Brownian motion, B(t)

    u(t) = a + b*t + 
           H(t-t_eq)*(B(t))

  our observation is 

    obs(t) = u(t) + d*sin(2*pi*t) + e*sin(4*pi*t) +  
                    f*cos(2*pi*t) + g*cos(4*pi*t) + 
                    h_i*H(t-t_i)
  '''
  assert (diff == 0) | (diff == 1)

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
  # brownian motion, 4 seasonal parameters, a secular velocity and
  # baseline displacement, plus however many jumps 
  M = 8 + J

  # return empty arrays if there are no observations
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

  # set initial prior covariance
  state_prior_cov[0,:,:] = init_prior_var*np.eye(M) 

  if prior_vel is not None:
    state_prior[:,3] = prior_vel
    state_prior_cov[:,3,3] = prior_vel_var

  # build system matrix for every time step
  # Brownian motion
  system[:,0] = _H(t-teq)
  # Brownian motion velocity
  system[:,1] = 0.0
  # baseline
  system[:,2] = 1.0
  # secular velocity
  system[:,3] = t
  # first annual seasonal term
  system[:,4] = np.sin(2*np.pi*t)
  # second annual seasonal term
  system[:,5] = np.cos(2*np.pi*t)
  # first semi-annual seasonal term
  system[:,6] = np.sin(4*np.pi*t)
  # second semi-annual seasonal term
  system[:,7] = np.cos(4*np.pi*t)
  for j,val in enumerate(jumps):
    system[:,8+j] = _H(t-val)

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
      out = _predict(trans[i+1],trans_cov[i+1],state_post[i],state_post_cov[i])
      state_prior[i+1] = out[0]
      state_prior_cov[i+1] = out[1]
    
  # smooth the state variables 
  state_smooth[-1] = state_post[-1]
  state_smooth_cov[-1] = state_post_cov[-1]

  for k in range(N-1)[::-1]:
    # dont bother smoothing data before the earthquake
    if (t[k] < teq) & detrend: 
      continue

    out = _rts_smooth(state_post[k],state_post_cov[k],
                      state_prior[k+1],state_prior_cov[k+1],
                      state_smooth[k+1],state_smooth_cov[k+1],
                      trans[k+1])
    state_smooth[k] = out[0]
    state_smooth_cov[k] = out[1]

  # the signal is assumed to be zero with high confidence before teq
  # state_smooth[t<teq,0] = 0.0
  # state_smooth[t<teq,1] = 0.0
  # state_smooth_cov[t<teq,0,0] = 1e-10
  # state_smooth_cov[t<teq,1,1] = 1e-10

  # check if all covariances matrices are positive definite. If not
  # then numerical errors may have significantly influenced the 
  # solution
  for i in range(N):
    # dont bother checking for positive definiteness before the
    # earthquake since that data will not be used
    if (t[i] < teq) & detrend:
      continue

    if not is_pos_def(state_smooth_cov[i]):
      print('WARNING: smoothed covariance at time %s is not positive '
            'definite. This may be due to an accumulation of numerical '
            'error. Consider rescaling the input parameters or '
            'changing the initial prior and covariance' % t[i])

  if detrend:
    state_smooth[t<teq] = 0.0
    state_smooth_cov[t<teq] = 1e-10
    out = state_smooth[:,diff]
    out_var = state_smooth_cov[:,diff,diff]

  else:
    # return the prediction to the data
    out = np.einsum('...i,...i',system,state_smooth)
    out_var = np.einsum('...i,...ij,...j',system,state_smooth_cov,system)

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
