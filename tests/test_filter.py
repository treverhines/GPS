import matplotlib.pyplot as plt
import numpy as np
import modest
import gps.filter
import scipy.special

def H(t):
  return (t>=0).astype(float)

# create true signal
N = 1000
t = np.linspace(10.3,11.0,N)

# the true signal which I am trying to recover consists of a linear trend with a rate of 0.05, 
# a step at 10.257 with magnitude 0.01, plus a logarithm term
utrue = 0.01*H(t-10.257) - 0.1*t  + 0.05*np.log(1 + H(t-10.257)*(t-10.257)/0.1)
veltrue = 0.05*np.log(1 + H(t-10.257)*(t-10.257)/0.1)
veltrue = np.diff(veltrue)/np.diff(t)

res = []

# create synthetic noise
sigma = 0.001
# add white noise
noise = np.random.normal(0.0,sigma,N) 
# add seasonal term
noise += 0.0*np.sin(2*np.pi*t+0.1) + 0.0*np.cos(4*np.pi*t+0.2)
# add a jump at 10.6
  
uobs = utrue + noise 

upred,uvar = gps.filter.stochastic_filter(uobs,sigma**2*np.ones(N),t,teq=10.257,
                                          jumps=[],diff=1,alpha=0.1,
                                          init_prior_var=0.1,detrend=True)

plt.figure(1)
plt.plot(t,uobs,'b-')
plt.fill_between(t,upred+np.sqrt(uvar),upred-np.sqrt(uvar),color='g',alpha=0.4)
plt.plot(t,upred,'g-')

upred,uvar = gps.filter.stochastic_filter(uobs,sigma**2*np.ones(N),t,teq=10.257,
                                          jumps=[],diff=1,alpha=0.1,prior_vel=-0.1,prior_vel_var=1e-10,
                                          init_prior_var=0.1,detrend=True)

plt.fill_between(t,upred+np.sqrt(uvar),upred-np.sqrt(uvar),color='r',alpha=0.4)
plt.plot(t,upred,'r-')

plt.plot(t[1:],veltrue,'m-')
plt.show()

