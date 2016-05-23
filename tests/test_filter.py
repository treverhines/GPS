import matplotlib.pyplot as plt
import numpy as np
import modest
import gps.filter
import scipy.special

def func_utrue(t):
  return np.log(t + 1.0)

def func_noise(t,sigma):
  return np.random.normal(0.0*sigma,sigma)


# create true signal
alpha = 0.5
N = 100
t1 = np.linspace(0.0,2.2,N//2)
t2 = np.linspace(2.21,5.5,N//2)
t = np.concatenate((t1,t2))
N = len(t)

sigma = 0.1*np.ones(N)
utrue = func_utrue(t)
noise = func_noise(t,sigma)
u = utrue + noise

upred,ucov = gps.filter.stochastic_filter(u,sigma**2,t,alpha=alpha)

fig,ax = plt.subplots()
ax.errorbar(t,u,sigma,0.0*sigma,fmt='.',color='k')
ax.plot(t,upred)
ax.fill_between(t,upred+np.sqrt(ucov),upred-np.sqrt(ucov),alpha=0.5,color='b')
ax.set_xlim((-1.0,6.0))

plt.show()
