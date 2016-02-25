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
#utrue = 0.01*H(t-10.257) + 0.02*t  + 0.01*np.log(1 + H(t-10.257)*(t-10.257)/0.1)
utrue = 0.01*H(t-10.257) + 0.02*t  + 0.01*(1 - np.exp(-H(t-10.257)*(t-10.257)/0.5))


# create synthetic noise
sigma = 0.00001
# add white noise
noise = np.random.normal(0.0,sigma,N) 
# add seasonal term
noise += 0.00001*np.sin(2*np.pi*t+0.1) + 0.0001*np.cos(4*np.pi*t+0.2)
# add a jump at 10.6
noise += 0.4*H(t-10.6)
  
uobs = utrue + noise 

upred,uvar = gps.filter.logexp_filter(uobs,sigma**2*np.ones(N),t,10.257,
                                          jumps=[10.6],diff=1,
                                          detrend=False)


plt.figure(2)
plt.plot(t[1:],np.diff(utrue)/(t[1]-t[0]),'m')
plt.plot(t,utrue,'b-')
plt.plot(t,uobs,'ko')
plt.fill_between(t,upred+np.sqrt(uvar),upred-np.sqrt(uvar),color='g',alpha=0.4)
plt.plot(t,upred,'g-')
plt.show()

