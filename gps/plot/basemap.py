#!/usr/bin/env python
from mpl_toolkits.basemap import Basemap as _Basemap
import numpy as np
#from netCDF4 import Dataset
from matplotlib.colors import LightSource
import matplotlib.cm
from pegtop import set_shade

class Basemap(_Basemap):
  def drawtopography(self,resolution=200,cmap=matplotlib.cm.gray,vmin=None,vmax=None,**kwargs):
    #try:
    #  url = '/cmld/data5/hinest/Data/SRTM/SoCal.h5'
    #  etopodata = Dataset(url)
    #except RuntimeError:
    url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo1.nc'
    etopodata = Dataset(url)

    lons = etopodata.variables['x'][:]
    lats = etopodata.variables['y'][:]
    # This leads to branch cut issues if dealing with datasets that go
    # over the prime meridian 
    xidx = (lons > self.lonmin) & (lons < self.lonmax)
    yidx = (lats > self.latmin) & (lats < self.latmax)
    lons = lons[xidx]
    lats = lats[yidx]

    topoin = etopodata.variables['rose'][yidx,xidx]
    topoin = np.ma.masked_array(topoin,np.isnan(topoin))
    topoin.data[topoin.mask] = 0.0

    nx = int((self.xmax-self.xmin)/resolution)+1
    ny = int((self.ymax-self.ymin)/resolution)+1
    topodat = self.transform_scalar(topoin,lons,lats,nx,ny,order=3,masked=True)

    ls = LightSource(azdeg = 270,altdeg = 45)
    #rgb = ls.shade(topodat,cmap)
    rgb = set_shade(topodat,cmap=cmap,scale=10.0,azdeg=300.0,altdeg=45.0,vmin=vmin,vmax=vmax)
    self.imshow(rgb,**kwargs)


  
