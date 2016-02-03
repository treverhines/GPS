import matplotlib.pyplot as plt
import numpy as np
import gps.filter
def H(t):
  return (t>=0).astype(float)

# create true signal
N = 365*10
t = np.linspace(-5.0,5.0,N)
utrue = 0.1*np.log(1 + t*H(t)/0.1)
utrue += 0.05*H(t-0.0)

# create synthetic correlated noise
cov = 0.001**2/(1 + 50.0*(t[:,None] - t[None,:])**2)
cov += 0.001**2*np.eye(N)
noise = np.random.multivariate_normal(np.zeros(N),cov,1)[0]


uobs_var = np.diag(cov)
uobs = utrue + noise

# add a jump to the noise at time 2 
uobs -= 0.01*H(t-2.0)


alpha = 0.01
jumps = [2.0]
restarts = [0.0]
upred,upred_var = gps.filter.stochastic_filter(uobs,uobs_var,t,alpha,jumps=jumps,restarts=restarts,init_prior_var=1e2)
vpred,vpred_var = gps.filter.stochastic_filter(uobs,uobs_var,t,alpha,jumps=jumps,restarts=restarts,diff=1,init_prior_var=1e2)
# add coore
plt.plot(t,0.01*np.log(1 + t*H(t)/0.1))
plt.plot(t,0.01*np.log(1 + t*H(t)/1.0))
plt.plot(t,0.01*np.log(1 + t*H(t)/10.0))
plt.plot(t,utrue)
plt.errorbar(t,uobs,np.sqrt(uobs_var),fmt='k.')
plt.errorbar(t,upred,np.sqrt(upred_var),fmt='b.')
#plt.errorbar(t,vpred,np.sqrt(vpred_var),fmt='r.')
plt.show()
