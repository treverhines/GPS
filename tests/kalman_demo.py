import matplotlib.pyplot as plt
import numpy as np
import modest
import gps.filter
import scipy.special

def stochastic_filter(u,var,t,alpha=0.1,init_prior_var=1.0):
  u = np.asarray(u)
  u = u[:,None]
  var = np.asarray(var)
  var = var[:,None,None]
  t = np.asarray(t)
  N = t.shape[0]
  M = 2

  state_prior = np.zeros((N,M))
  state_post = np.zeros((N,M))
  state_prior_cov = np.zeros((N,M,M))
  state_post_cov = np.zeros((N,M,M))
  state_smooth = np.zeros((N,M))
  state_smooth_cov = np.zeros((N,M,M))
  trans = np.zeros((N,M,M))
  trans_cov = np.zeros((N,M,M))
  system = np.zeros((N,M))

  state_prior_cov[0,:,:] = init_prior_var*np.eye(M)

  system[:,0] = 1.0
  system[:,1] = 0.0

  dt = np.diff(t)
  trans[1:] = np.eye(M)
  trans[1:,0,1] = dt
  trans_cov[1:,0,0] = alpha**2*0.333*dt**3
  trans_cov[1:,0,1] = alpha**2*0.500*dt**2
  trans_cov[1:,1,0] = alpha**2*0.500*dt**2
  trans_cov[1:,1,1] = alpha**2*dt

  for i in range(N):
    out = gps.filter._update(system[[i]],u[i],var[i],state_prior[i],state_prior_cov[i])
    state_post[i] = out[0]
    state_post_cov[i] = out[1]
    if i != (N-1):
      out = gps.filter._predict(trans[i+1],trans_cov[i+1],state_post[i],state_post_cov[i])
      state_prior[i+1] = out[0]
      state_prior_cov[i+1] = out[1]

  state_smooth[-1] = state_post[-1]
  state_smooth_cov[-1] = state_post_cov[-1]

  for k in range(N-1)[::-1]:
    out = gps.filter._rts_smooth(state_post[k],state_post_cov[k],
                      state_prior[k+1],state_prior_cov[k+1],
                      state_smooth[k+1],state_smooth_cov[k+1],
                      trans[k+1])
    state_smooth[k] = out[0]
    state_smooth_cov[k] = out[1]

  for i in range(N):
    if not gps.filter.is_pos_def(state_smooth_cov[i]):
      print('WARNING: smoothed covariance at time %s is not positive '
            'definite. This may be due to an accumulation of numerical '
            'error. Consider rescaling the input parameters or '
            'changing the initial prior and covariance' % t[i])

  return state_prior,state_prior_cov,state_post,state_smooth

np.random.seed(5)

# create true signal
N = 30
t = np.linspace(0,10.0,N)

# the true signal which I am trying to recover consists of a linear trend with a rate of 0.05, 
# a step at 10.257 with magnitude 0.01, plus a logarithm term
utrue = 10*np.log(t + 1)
sigma = 2*np.ones(N)
noise = np.random.normal(0,2,N)
uobs = utrue + noise
alpha = 50.0
upreds = stochastic_filter(uobs,sigma**2,t,alpha=alpha,init_prior_var=100.0)
uprior,uprior_cov,upost,smooth = upreds

fig,ax = plt.subplots(figsize=(7,5))
ax.plot(t,utrue,'b--',lw=2,alpha=0.5)
ax.text(3.65,4.5,'velocity variance: %s (mm/yr)$^2$/yr' % alpha)
ax.errorbar(t,uobs,sigma,fmt='ko')
ax.plot(t,smooth[:,0],'b-',lw=2)
ax.set_xlim((-0.5,10.5))
ax.set_ylim((-2.0,27.0))
ax.set_xlabel('time (years)')
ax.set_ylabel('displacement (mm)')
ax.set_title('Kalman filter demonstration')
ax.legend(['true signal','observed','filtered'],numpoints=3,loc=8,fontsize=12,frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.show()



