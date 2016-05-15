#!/usr/bin/env python
import gps
import gps.plot
import gps.station
import numpy as np
import matplotlib.pyplot as plt
import logging
import h5py

logging.basicConfig(level=logging.DEBUG)

#db = gps.station.StationDB('pos_sample')
#time = np.arange(db.meta['start'],db.meta['end'],db.meta['min_tolerance'])
#dat = db.get_data_array(time)
f = h5py.File('/cmld/data5/hinest/PSGI/ElMayorFinal/data/data_5.0yr.h5','r')
dat = {}
dat['mean'] = f['displacement']['mean'][...]
dat['covariance'] = f['displacement']['covariance'][...]
dat['variance'] = f['displacement']['variance'][...]
dat['time'] = f['displacement']['time'][...] + 2010.257
dat['position'] = f['displacement']['position'][...]
dat['name'] = f['displacement']['name'][...]
dat['mask'] = f['displacement']['mask'][...]

dat2 = {}
dat2['mean'] = f['velocity']['mean'][...]
dat2['covariance'] = f['velocity']['covariance'][...]
dat2['variance'] = f['velocity']['variance'][...]
dat2['time'] = f['velocity']['time'][...] + 2010.257
dat2['position'] = f['velocity']['position'][...]
dat2['name'] = f['velocity']['name'][...]
dat2['mask'] = f['velocity']['mask'][...]
dat2['mask'][dat2['time']>2013,:] = True

pazimuth = (90 - 357.0) * np.pi/180 
angle_range = (pazimuth - 10*np.pi/180,pazimuth + 10*np.pi/180)
gps.plot.record_section([dat,dat2],(-115.37,32.35),(10000,380000),angle_range)

