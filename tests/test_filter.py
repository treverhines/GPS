import matplotlib.pyplot as plt
import numpy as np
import modest
import gps.filter
import scipy.special

def H(t):
  return (t>=0).astype(float)

# create true signal
N = 1000
t = np.linspace(8.7,11.0,N)

# the true signal which I am trying to recover consists of a linear trend with a rate of 0.05, 
# a step at 10.257 with magnitude 0.01, plus a logarithm term
utrue = 0.01*H(t-10.257) + 0.02*t  + 0.01*np.log(1 + H(t-10.257)*(t-10.257)/0.1)

res = []
for i in range(100):
  # create synthetic noise
  sigma = 0.001
  # add white noise
  noise = np.random.normal(0.0,sigma,N) 
  # add seasonal term
  noise += 0.005*np.sin(2*np.pi*t+0.1) + 0.04*np.cos(4*np.pi*t+0.2)
  # add a jump at 10.6
  noise += 0.4*H(t-10.6)
  
  uobs = utrue + noise 
  upred,uvar = gps.filter.stochastic_filter(uobs,sigma**2*np.ones(N),t,teq=10.257,
                                          jumps=[10.6],diff=0,alpha=0.1,
                                          init_prior_var=0.01,detrend=False)
  res += list((upred-utrue)/np.sqrt(uvar))

plt.figure(1)
plt.hist(res,50)
plt.figure(2)
plt.plot(t,utrue,'b-')
plt.fill_between(t,upred+np.sqrt(uvar),upred-np.sqrt(uvar),color='g',alpha=0.4)
plt.plot(t,upred,'g-')
plt.show()

