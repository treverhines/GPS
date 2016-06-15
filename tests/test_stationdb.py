#!/usr/bin/env python
import numpy as np
import gps.station
import logging
import matplotlib.pyplot as plt 

logger = logging.basicConfig(level=logging.INFO)
db = gps.station.StationDB('pos_sample')

k1 = db.keys()[0]
print(k1)


t,u,c = db[k1][:]
ti = np.arange(db[k1].meta['start'],db[k1].meta['end'],1.0/365.25)
ui,ci = db[k1](ti)

plt.plot(t,u,'ko')
plt.plot(ti,ui,'bo')
plt.show()
#db.view()
