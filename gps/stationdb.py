#!/usr/bin/env python
import os
import urllib
import time  as timemod
import numpy as np
import datetime
import dateutil.parser
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse
import h5py
from matplotlib.widgets import Slider
import misc
import pandas
import logging
import scipy.interpolate
import gps.plot.plot_stationdb as plot_stationdb
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

TODAY = misc.decyear(*tuple(timemod.gmtime())[:3]) #today in decimal year (eg. 2014.125)

try:
  GPSPATH = os.environ['GPSPATH']
except KeyError:
  print('\ncannot find GPSPATH: '
        'make an empty directory to store data downloaded from ' 
        'unavco.org and save that directory location in your ' 
        'environment as "GPSPATH"\n')
  
def nearest(x):
  '''   
  returns a list of distances to the nearest point for each point in   
  x.
  '''
  x = np.asarray(x)
  if len(np.shape(x)) == 1:
    x = x[:,None]

  N = len(x)
  A = (x[None] - x[:,None])**2
  A = np.sqrt(np.sum(A,2))
  A[range(N),range(N)] = np.max(A)
  nearest_dist = np.min(A,1)
  return nearest_dist


def decyear(datetime):
  return misc.decyear(datetime.year,
                      datetime.month,
                      datetime.day,
                      datetime.hour,
                      datetime.minute)


def decyear_inv(t):
  date_iso = misc.decyear_inv(t)
  date = dateutil.parser.parse(date_iso)
  return date


def decyear_range(datetime1,datetime2,delta=None):
  if delta is None:
    delta = datetime.timedelta(1)

  now = datetime1
  out = []
  while True:
    if now < datetime2:
      out += [decyear(now)]
      now += delta

    else:
      return out  


def swap_axes(m,ax1,ax2):
  mshape = np.shape(m)
  axes = len(mshape)
  if ax1 == -1:
    ax1 = axes - 1

  if ax2 == -1:
    ax2 = axes - 1

  indices = range(axes)
  indices.pop(ax1)
  indices.insert(ax1,ax2)
  indices.pop(ax2)
  indices.insert(ax2,ax1)
  out = np.einsum(m,indices)  
  out.flags['WRITEABLE'] = True
  return out


class MaskedInterp1D(interp1d):
  '''
  Interpolation class which inherits from `interp1d` and returns
  masked arrays for invalid interpolation points. Invalid points are
  either outside the bounds of the provided x values or are too far
  away from the provided x values
  '''
  def __init__(self,x,y,bounds_error=False,**kwargs):
    interp1d.__init__(self,x,y,
                      bounds_error=bounds_error,
                      **kwargs)

  def __call__(self,x,tol=np.inf):
    itp = interp1d.__call__(self,x)
    itp = np.ma.array(itp,mask=self.invalid_mask(x,tol))
    return itp

  def invalid_mask(self,x,tol):
    isscalar = np.isscalar(x)
    nearest = np.min(np.abs(self.x[:,None] - x),0)
    if isscalar:
      nearest = nearest[0]

    x_mask = np.asarray((nearest > tol) | 
                        (x < self.x[0]) | 
                        (x > self.x[-1]),dtype=int)

    itp_shape = list(np.shape(self.y))
    itp_shape.pop(self.axis)
    if isscalar:
      itp_mask = np.zeros(itp_shape,dtype=int)
      itp_mask += x_mask     
      return itp_mask
      
    else:
      itp_shape.insert(self.axis,len(x))
      itp_mask = np.zeros(itp_shape,dtype=int)
      # reshape the interpolation mask so that the x mask can 
      # be broadcasted on it 
      #itp_mask = swap_axes(itp_mask,self.axis,-1) 
      itp_mask = np.swapaxes(itp_mask,self.axis,-1) 
      itp_mask += x_mask     
      #itp_mask = swap_axes(itp_mask,self.axis,-1) 
      itp_mask = np.swapaxes(itp_mask,self.axis,-1) 
      return itp_mask


class StationDB(dict):
  '''
  Stores, filters, and plots data downloaded from unavco.org

  behaves like a dictionary where each key is a station ID and the associated
  values are the station metadata or displacements in a particular direction

  EXAMPLE USAGE:

  >> stationdb = StationDB(selection_type='circle',
                           center=[32.258,-115.289]
                           repository='ElMayor')
  >> stationdb['P495']['north']['raw'] % calls uncleaned northing displacements
  >> stationdb.condense(0.1,cut_times=[2010.25]) % filters and subsamples 
  >> stationdb.detrend(2010.25) % detrends data at April 4, 2010
  >> stationdb.view() % displays data

  To do: -modify the hierarchy so that disp_type comes before disp_direction. 
         -fix "add_displacment" so that the new times do not need to match 
          existing times
  '''
  def __init__(self,repository):
    '''
    PARAMETERS
    ----------
      selection_type: Excludes stations within the data repository from being 
                      included in this StationDB instance.  Can be either 'box'
                      (excludes stations outside a bounding box), 'circle'
                      (excludes stations outside a bounding circle), or 'all'
                      (no stations are excluded)
      center: center of either the bounding box or bounding circle 
              (latitude,longitude) (not needed 'all' is specified)
      radius: radius of the bounding circle in meters
      dx: half-width (E-W direction) of the bounding box in meters
      dy: half-height (N-S direction) of the bounding box in meters
      basemap: basemap to be used when view() is called.  A basemap is created
               which includes all stations in the repository is no basemap is 
               provided
      time_range: Displacements measured within this time range will be 
                  downloaded
      repository: Name of directory where data is downloaded (created and 
                  populated with the update_data function
    '''  
    self.repository = repository
    #self.metadata = pandas.io.parsers.read_csv(
    #                  '%s/%s/metadata' % (GPSPATH,repository),
    #                  skiprows=[0],
    #                  index_col=[0],
    #                  names=['Description','Latitude','Longitude'])
    metadata = pandas.io.parsers.read_csv(
                 '%s/%s/metadata' % (GPSPATH,repository),
                 skiprows=[0],
                 names=['name','description','latitude','longitude'])
    # turn metadata into a dictionary because a pandas array is 
    # not necessary 
    name = np.array(metadata['name'])
    description = np.array(metadata['description'])
    longitude = np.array(metadata['longitude'])
    latitude = np.array(metadata['latitude'])

    self.metadata = {'stations':0,
                     'start':np.inf,
                     'end':-np.inf}
    
    for n,d,lon,lat in zip(name,description,longitude,latitude):
      logger.debug('initiating station %s' % n)
      self[n] = {'description':d,
                 'longitude':lon,
                 'latitude':lat}
      self._add_station(n)

      self.metadata['stations'] += 1

      if self.metadata['start'] > self[n]['start']:
        self.metadata['start'] = self[n]['start']

      if self.metadata['end'] < self[n]['end']:
        self.metadata['end'] = self[n]['end']
  
    return

  def remove_station(self,name):
    self.pop(name)


  def _add_station(self,name):
    fname = '%s/%s/%s' % (GPSPATH,self.repository,name)
    value_names = ['E','N','V',
                   'EStd','NStd',
                   'VStd','ENCor',
                   'EVCor','NVCor']
    data = pandas.io.parsers.read_csv(
             fname,skiprows=1,index_col=[0],parse_dates=True,
             names=value_names) 
    time = data.index.map(decyear)
    value = [data['E'],data['N'],data['V']]
    value = np.einsum('ij->ji',value)
  
    cov = [[data['EStd']**2,
            data['ENCor']/(data['EStd']*data['NStd']),
            data['EVCor']/(data['EStd']*data['VStd'])],
           [data['ENCor']/(data['EStd']*data['NStd']),
            data['NStd']**2,
            data['NVCor']/(data['NStd']*data['VStd'])],
           [data['EVCor']/(data['EStd']*data['VStd']),
            data['NVCor']/(data['NStd']*data['VStd']),
            data['VStd']**2]]
    cov = np.einsum('ijk->kij',cov)

    self[name]['value'] = MaskedInterp1D(time,value,
                                         axis=0,
                                         kind='nearest')

    self[name]['covariance'] = MaskedInterp1D(time,cov,
                                              axis=0,
                                              kind='nearest')

    self[name]['start'] = self[name]['value'].x[0]

    self[name]['end'] = self[name]['value'].x[-1]

  def __repr__(self):
    '''
    eventually change this so that is returns a useful summary
    '''
    return object.__repr__(self)


  def data_array(self,date1=None,date2=None,delta=None):
    if date1 is None:
      date1 =  decyear_inv(self.metadata['start'])

    if date2 is None:
      date2 =  decyear_inv(self.metadata['end'])

    if np.isscalar(delta):
      delta = datetime.timedelta(delta)

    if delta is None:
      delta = datetime.timedelta(1)

    times = decyear_range(date1,date2,delta)
    times = np.array(times) 
    # find distance to next nearest time
    times_diff = nearest(times)/2.0
    names = self.keys()
    lon = [self[n]['longitude'] for n in names]
    lat = [self[n]['latitude'] for n in names]
    data = np.ma.zeros((len(times),len(names),3))
    covariance = np.ma.zeros((len(times),len(names),3,len(names),3))
    for i,n in enumerate(names):
      data[:,i,:] = self[n]['value'](times,tol=times_diff)
      covariance[:,i,:,i,:] = self[n]['covariance'](times,tol=times_diff)

    return data,covariance,times,names,lon,lat


  def write(self,name,date1=None,date2=None,delta=None):
    f = h5py.File(name,'w-')
    d,c,t,n,lon,lat = self.data_array(date1,date2,delta)
    f['mean'] = d.data
    f['mean_mask'] = d.mask
    f['covariance'] = c
    f['covariance_mask'] = c.mask
    f['position'] = np.array([lon,lat]).transpose()
    f['name'] = self.keys()
    f['time'] = t
    f.close()


  def view(self,date1=None,date2=None,delta=None,**kwargs):
    '''
    Display GPS data in two figures. 1) a map view of displacement vectors, and
    2) a time series plot for selected stations

    PARAMETERS
    ----------
      disp_type: The data types that will be plotted if available.  Must be a
                 vector with elements 'raw','secular','detrended','predicted',
                 or 'residual'.

                 'raw': observered displacements that are not detrended
                 'secular': best fitting secular trend (available after calling
                            the 'detrend' method)
                 'detrended': 'raw' minus 'secular' displacements (available 
                              after calling the 'detrend' method)
                 'predicted': predicted displacements (available after calling 
                              the 'add_predicted' method
                 'residual': 'detrended' minus 'predicted' displacements 
                             (available after calling the 'add_predicted' 
                             method)
      ts_formats: vector of time series line formats for each disp type
                 (e.g. ['ro','b-'])                 
      quiver_colors: vector of quiver arrow colors for each disp type
      ref_time: If provided, zeros each of the displacement types at this time
      time_range: Range of times for the time series plot and map view time
                  slider
      scale_length: real length (in mm) of the map view scale
      quiver_scale: changes lengths of the quiver arrows, 
                    smaller number -> bigger arrow
      artists: additional artists to add to the map view plot
    '''
    out = self.data_array(date1,date2,delta)
    plot_stationdb.view([out[0]],
                        [out[1]],
                        out[2],
                        out[3],
                        out[4],
                        out[5],
                        **kwargs)

##----------------------------------------------------------------------
def update_data(lat_range,lon_range,repository='data'):
  '''
  Downloads the most recent GPS data from unavco.org
 
  PARAMETERS
  ----------
    lat_range: GPS data will be downloaded for stations within this range of
               latitudes
    lon_range: GPS data will be downloaded for stations within this range of
               longitudes
    repository: The GPS timeseries files and metadata file with be stored in
                this directory.  The full path to the downloaded data will be
                '<GPSPATH>/<repository>'

  HARDCODED PARAMETERS
  --------------------
    reference frame: NAM08 
    start of data acquisition: January 1, 2000
    end of data acquisition: today
  '''  
  if os.path.exists('%s/%s' % (GPSPATH,repository)):
    sp.call(['rm','-r','%s/%s' % (GPSPATH,repository)])
  os.mkdir('%s/%s' % (GPSPATH,repository))
  ref_frame            = 'nam08'
  start_time           = '2000-01-01T00:00:00'
  end_time             = timemod.strftime('%Y-%m-%dT00:00:00')
  string               = ('http://web-services.unavco.org:80/gps/metadata/'
                          'sites/v1?minlatitude=%s&maxlatitude='
                          '%s&minlongitude=%s&maxlongitude=%s&limit=10000' %
                          (lat_range[0],
                           lat_range[1],
                           lon_range[0],
                           lon_range[1]))
  buff                 = urllib.urlopen(string) 
  metadata_file        = open('%s/%s/metadata.txt' % (GPSPATH,repository),'w')
  file_string          = buff.read()
  buff.close()
  file_lst             = file_string.strip().split('\n')
  station_lst          = []
  for i in file_lst[1:]:
    lst                = i.split(',')
    ID                 = lst[0]
    description        = lst[1]
    lat                = float(lst[2])
    lon                = float(lst[3]) 
    string             = ('http://web-services.unavco.org:80/gps/data/position'
                          '/%s/v1?referenceFrame=%s&starttime=%s&endtime='
                          '%s&tsFormat=iso8601' %
                         (ID,ref_frame,start_time,end_time))
    buff               = urllib.urlopen(string)  
    station_exists     = any([j[0] == ID for j in station_lst])
    if station_exists:
      logger.warning('station_exists')
    if ((buff.getcode() != 404) & 
        (not station_exists)):
      logger.info('updating station %s (%s)' % (ID,description))
      file_string      = buff.read()
      out_file         = open('%s/%s/%s' % (GPSPATH,repository,ID),'w')
      out_file.write(file_string)
      out_file.close()      
      station_lst     += [(ID,description,lat,lon)]
      metadata_file.write('%s,%s,%s,%s\n' % (ID,description,lat,lon))
    metadata_file.flush()
    buff.close()   
  metadata_file.close()
  return




