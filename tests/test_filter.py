import matplotlib.pyplot as plt
import numpy as np
import modest
import gps.filter
import scipy.special
def H(t):
  return (t>=0).astype(float)


def remove_jumps_and_seasonals(u,var,t,jumps):
  J = len(jumps)
  def system(m,t):
    out = np.zeros(len(t))
    out += m[0]
    out += m[1]*t
    out += m[2]*np.sin(2*np.pi*t)
    out += m[3]*np.sin(4*np.pi*t)
    out += m[4]*np.cos(2*np.pi*t)
    out += m[5]*np.cos(4*np.pi*t)
    for i in range(J):
      out += m[2+i]*H(t-jumps[i])

    return out
      
  signal_indices = np.array([0,1])
  jacobian = modest.make_jacobian(system)  

  m,mcov = modest.nonlin_lstsq(system,u,2+J,data_covariance=var,output=['solution','solution_covariance'],system_args=(t,))
  jac = jacobian(m,t)
  m = m[signal_indices]
  jac = jac[:,signal_indices]
  mcov = mcov[np.ix_(signal_indices,signal_indices)] 
  pred = jac.dot(m)
  predcov = jac.dot(mcov).dot(jac.T)
  predvar = np.diag(predcov)
  plt.errorbar(t,u,np.sqrt(var))
  plt.errorbar(t,pred,np.sqrt(predvar))
  plt.show()


# create true signal
N = 100
t = np.linspace(-2.5,4.5,N)

utrue = 0.0 + 1.0*t + 1.0*H(t) + 0.0*H(t)*t + np.log(1 + H(t)*t/0.5) + np.sin(2*np.pi*t)

# create synthetic correlated noise
cov = 0.1**2*np.eye(N)
#noise += 0.001*np.sin(2*np.pi*(t-0.5)) + 0.001*np.cos(4*np.pi*(t-0.17)) 
#noise -= 0.01*H(t-2.0) 
var = np.diag(cov)
uobs = utrue# + noise 
#upred,uvar = gps.filter.log_filter(uobs,var,t,0.0,[])
upred,uvar = gps.filter.log_filter(uobs,var,t,0.0,[2.0],diff=1)

plt.plot(t,utrue,'b-')
plt.errorbar(t,uobs,np.sqrt(var),fmt='k.')
plt.errorbar(t,upred,np.sqrt(uvar),fmt='g-')
plt.plot(t,1 + 1.0/(t+0.5))
plt.show()
quit()
#remove_jumps_and_seasonals(uobs,var,t,[])

#upred = modest.nonlin_lstsq(system,utrue,6,output=['predicted'],system_args=(t,))
plt.plot(t,utrue,'k')
plt.plot(t,upred,'b')
plt.show()




uobs_var = np.diag(cov)
uobs = utrue + noise


plt.plot(t,utrue)
plt.plot(t,uobs)
plt.plot(t,upred)
plt.show()
quit()
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
