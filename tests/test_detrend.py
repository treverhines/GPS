#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import gps.filter

Nt = 100
t = np.linspace(0.0,10.0,Nt)
u = 2.0 + 3.0*t
u += np.sin(2*np.pi*t)
u += np.random.normal(0.0,0.1,Nt)

out = gps.filter.stochastic_detrender(u,0.1**2*np.ones(Nt),t,teq=5.0,detrend=True)
uout = out[0]
sout = np.sqrt(out[1])

fig,ax = plt.subplots()
ax.plot(t,u,'k-')
ax.plot(t,uout,'k-')
ax.fill_between(t,uout-sout,uout+sout)
plt.show()



