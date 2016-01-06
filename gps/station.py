#!/usr/bin/env python
from __future__ import division
import numpy as np
import datetime
import time 
import re
import os
import logging
import conversions
import modest
import maskitp
import collections
import h5py
import plot

logger = logging.getLogger(__name__)


def find_field(field,s):
  out = re.search('%s\s*:\s*(.*?)\n' % field,s,re.DOTALL+re.IGNORECASE)
  if out is None:
    logger.warning('no match found for field "%s"' % field)
    return None
  else:
    return out.group(1)


def string_to_decyear(s,fmt):
  d = datetime.datetime.strptime(s,fmt)

  date_tuple = d.timetuple()
  # time in seconds of d
  time_in_sec = time.mktime(date_tuple)

  date_tuple = datetime.datetime(d.year,1,1,0,0).timetuple()
  # time in seconds of start of year
  time_year_start = time.mktime(date_tuple)

  date_tuple = datetime.datetime(d.year+1,1,1,0,0).timetuple()
  # time in seconds of start of next year
  time_year_end = time.mktime(date_tuple)
  decimal_time    = (d.year + (time_in_sec - time_year_start)/
                      (time_year_end - time_year_start))

  return decimal_time


def parse_pos_string(string):
  # output metadata dictionary
  metadata = {'name':None,
              'id':None,
              'start':None,
              'end':None,
              'reference_frame':None,
              'longitude':None,
              'latitude':None,
              'height':None,
              'observations':None}

  # intermediate dictionary used to help parse through pos file
  data_ = {'time':None,
           'east disp.':None,
           'north disp.':None,
           'vert. disp.':None,
           'east std. dev.':None,
           'north std. dev.':None,
           'vert. std. dev.':None,
           'north-east corr.':None,
           'north-vert. corr.':None,
           'east-vert. corr.':None}

  # output data dictionary
  data = {'time':None,
          'mean':None,  
          'cov':None}

  version = find_field('Format Version',string)
  if version == '1.1.0':
    metadata['reference_frame'] = find_field('reference frame',string)
    metadata['name'] = find_field('name',string)
    metadata['id'] = find_field('4-character ID',string)
    start_time_string = find_field('First Epoch',string)
    metadata['start'] = string_to_decyear(start_time_string,'%Y%m%d %H%M%S')
    end_time_string = find_field('Last Epoch',string)
    metadata['end'] = string_to_decyear(end_time_string,'%Y%m%d %H%M%S')
    ref_pos_string =  find_field('NEU Reference position',string)
    ref_pos = [float(i) for i in ref_pos_string.strip().split()[:3]]
    lon,lat = conversions.bound_lon_lat(ref_pos[1],ref_pos[0]) 
    metadata['longitude'] = lon
    metadata['latitude'] = lat
    metadata['height'] = ref_pos[2]
    
    data_string = re.search(r'\*(.*)',string,re.DOTALL).group(1)

    data_array = [i.strip().split() for i in data_string.strip().split('\n')]
    data_array = np.array(data_array,dtype=str)

    data_headers = data_array[0,:]
    data_values = data_array[1:,:]

    date_idx = np.nonzero(data_headers=='YYYYMMDD')[0][0]
    time_idx = np.nonzero(data_headers=='HHMMSS')[0][0]

    DN_idx = np.nonzero(data_headers=='dN')[0][0]
    DE_idx = np.nonzero(data_headers=='dE')[0][0]
    DU_idx = np.nonzero(data_headers=='dU')[0][0]

    Sn_idx = np.nonzero(data_headers=='Sn')[0][0]
    Se_idx = np.nonzero(data_headers=='Se')[0][0]
    Su_idx = np.nonzero(data_headers=='Su')[0][0]

    Rne_idx = np.nonzero(data_headers=='Rne')[0][0]
    Rnu_idx = np.nonzero(data_headers=='Rnu')[0][0]
    Reu_idx = np.nonzero(data_headers=='Reu')[0][0]

    # turn the first two columns into a list of decyear times
    time = data_values[:,[date_idx,time_idx]]
    time = [''.join(i) for i in time]
    time = [string_to_decyear(i,'%Y%m%d%H%M%S') for i in time]
    metadata['observations'] = len(time)

    data_['north disp.'] = np.array(data_values[:,DN_idx],dtype=float)
    data_['east disp.'] = np.array(data_values[:,DE_idx],dtype=float)
    data_['vert. disp.'] = np.array(data_values[:,DU_idx],dtype=float)

    data_['north std. dev.'] = np.array(data_values[:,Sn_idx],dtype=float)
    data_['east std. dev.'] = np.array(data_values[:,Se_idx],dtype=float)
    data_['vert. std. dev.'] = np.array(data_values[:,Su_idx],dtype=float)

    data_['north-east corr.'] = np.array(data_values[:,Rne_idx],dtype=float)
    data_['north-vert. corr.'] = np.array(data_values[:,Rnu_idx],dtype=float)
    data_['east-vert. corr.'] = np.array(data_values[:,Reu_idx],dtype=float)

    mean = np.array([data_['east disp.'],
                     data_['north disp.'],
                     data_['vert. disp.']])
    mean = np.einsum('ij->ji',mean)

    cov = [[data_['east std. dev.']**2,
            data_['north-east corr.']*data_['east std. dev.']*data_['north std. dev.'],
            data_['east-vert. corr.']*data_['east std. dev.']*data_['vert. std. dev.']],
           [data_['north-east corr.']*data_['east std. dev.']*data_['north std. dev.'],
            data_['north std. dev.']**2,
            data_['north-vert. corr.']*data_['north std. dev.']*data_['vert. std. dev.']],
           [data_['east-vert. corr.']*data_['east std. dev.']*data_['vert. std. dev.'],
            data_['north-vert. corr.']*data_['north std. dev.']*data_['vert. std. dev.'],
            data_['vert. std. dev.']**2]]
    cov = np.einsum('ijk->kij',cov)

    data['time'] = np.array(time,copy=True)
    data['time'].setflags(write=False)

    data['mean'] = np.array(mean,copy=True)
    data['mean'].setflags(write=False)

    data['cov'] = np.array(cov,copy=True)
    data['cov'].setflags(write=False)

  else:
    logger.warning('version not recognized')

  return metadata,data

class Station:
  def __init__(self,pos_file_name,tol=1.0/365,db=None):
    f = open(pos_file_name,'r')
    pos_string = f.read()
    f.close()
    meta,data = parse_pos_string(pos_string)
    meta['tolerance'] = tol

    self.db = db
    self.itp = {}
    self.meta = meta 
    self.data = data
    self.itp['mean'] = maskitp.MaskedNearestInterp(
                         self.data['time'],
                         self.data['mean'],
                         self.meta['tolerance'])
    self.itp['cov'] = maskitp.MaskedNearestInterp(
                         self.data['time'],
                         self.data['cov'],
                         self.meta['tolerance'])

  def __repr__(self):
    string = '\nStation\n'
    string += '  id: %s (%s)\n' % (self.meta['id'],self.meta['name']) 
    string += '  time range: %.2f to %.2f\n' % (self.meta['start'],self.meta['end']) 
    string += '  observations: %s\n' % self.meta['observations']
    string += '  reference frame: %s\n' % self.meta['reference_frame']
    string += '  longitude: %.2f\n' % self.meta['longitude']
    string += '  latitude: %.2f\n' % self.meta['latitude']
    string += '  height: %.2f m\n' % self.meta['height']
    string += '  tolerance: %.4f years (%.1f days)' % (
                 self.meta['tolerance'],
                 self.meta['tolerance']*365)
    return string


  def set_metadata(self,**kwargs):
    self.meta.update(kwargs)

    if self.db is not None:
      self.db._update_metadata()      


  def set_data(self,time=None,mean=None,cov=None):
    if time is None:
      time = self.data['time']

    if mean is None:
      mean = self.data['mean']

    if cov is None:
      cov = self.data['cov']

    assert np.shape(time)[0] == np.shape(mean)[0]
    assert np.shape(time)[0] == np.shape(cov)[0]
    assert np.shape(mean)[1] == 3
    assert np.shape(cov)[1] == 3
    assert np.shape(cov)[2] == 3

    self.data['time'] = np.array(time,copy=True)
    self.data['time'].setflags(write=False)

    self.data['mean'] = np.array(mean,copy=True)
    self.data['mean'].setflags(write=False)

    self.data['cov'] = np.array(cov,copy=True)
    self.data['cov'].setflags(write=False)

    self.itp['mean'] = maskitp.MaskedNearestInterp(
                         self.data['time'],
                         self.data['mean'],
                         self.meta['tolerance'])
    self.itp['cov'] = maskitp.MaskedNearestInterp(
                         self.data['time'],
                         self.data['cov'],
                         self.meta['tolerance'])

    self.meta['start'] = np.min(self.data['time'])    
    self.meta['end'] = np.max(self.data['time'])    
    self.meta['observations'] = len(self.data['time'])

    if self.db is not None:
      self.db._update_metadata()      

  def get_data(self):
    return (np.copy(self.data['time']),
            np.copy(self.data['mean']),
            np.copy(self.data['cov']))


  def __call__(self,t):
    return self.itp['mean'](t),self.itp['cov'](t)        


  def __getitem__(self,i):
    return self.data['time'][i],self.data['mean'][i],self.data['cov'][i]        


class StationDB(collections.OrderedDict):
  def __init__(self,database_directory,tol=1.0/365):
    collections.OrderedDict.__init__(self)
    pos_files = os.listdir(database_directory)
    # remove files without a .pos extension
    pos_files = [i for i in pos_files if i[-4:] == '.pos']
    # sort stations alphabetically
    pos_files = np.sort(pos_files)
    pos_files = [database_directory+'/'+i for i in pos_files]
    self.meta = {'start':np.inf,
                 'end':-np.inf,
                 'reference_frame':None,
                 'stations':0,
                 'observations':0,
                 'min_longitude':180,
                 'max_longitude':-180,
                 'min_latitude':90,
                 'max_latitude':-90,
                 'min_tolerance':np.inf}

    for i in pos_files:
      sta = Station(i,tol=tol,db=self)
      logger.info('initializing station %s:%s' % (sta.meta['id'],sta.__repr__()))
      self[sta.meta['id']] = sta

    self._update_metadata()
        

  def pop(self,*args,**kwargs):
    collections.OrderedDict.pop(self,*args,**kwargs)
    self._update_metadata()

  def popitem(self,*args,**kwargs):
    collections.OrderedDict.popitem(self,*args,**kwargs)
    self._update_metadata()

  def update(self,*args,**kwargs):
    collections.OrderedDict.update(self,*args,**kwargs)
    self._update_metadata()

  def _update_metadata(self):
    self.meta = {'start':np.inf,
                 'end':-np.inf,
                 'reference_frame':None,
                 'stations':0,
                 'observations':0,
                 'min_longitude':180,
                 'max_longitude':-180,
                 'min_latitude':90,
                 'max_latitude':-90,
                 'min_tolerance':np.inf}

    for key,sta in self.iteritems():
      if not isinstance(sta,Station):
        logger.warning('item with key %s is not a Station instance' % key)
        continue

      if self.meta['reference_frame'] is None:
        self.meta['reference_frame'] = sta.meta['reference_frame']

      if self.meta['reference_frame'] != sta.meta['reference_frame']:
        logger.warning(
          'reference frame for station %s, %s, is not the same as '
          'previously added stations. Remove station with "pop" '
          'method' % (sta.meta['id'],sta.meta['reference_frame']))

      if self.meta['start'] > sta.meta['start']:
        self.meta['start'] = sta.meta['start']

      if self.meta['end'] < sta.meta['end']:
        self.meta['end'] = sta.meta['end']

      if self.meta['min_tolerance'] > sta.meta['tolerance']:
        self.meta['min_tolerance'] = sta.meta['tolerance']

      if self.meta['min_longitude'] > sta.meta['longitude']:
        self.meta['min_longitude'] = sta.meta['longitude']

      if self.meta['min_latitude'] > sta.meta['latitude']:
        self.meta['min_latitude'] = sta.meta['latitude']

      if self.meta['max_longitude'] < sta.meta['longitude']:
        self.meta['max_longitude'] = sta.meta['longitude']

      if self.meta['max_latitude'] < sta.meta['latitude']:
        self.meta['max_latitude'] = sta.meta['latitude']

      self.meta['observations'] += sta.meta['observations']
      self.meta['stations'] += 1


  def __repr__(self):
    string = '\nStationDB\n'
    string += '  time range: %.2f to %.2f\n' % (self.meta['start'],self.meta['end']) 
    string += '  stations: %s\n' % self.meta['stations']
    string += '  observations: %s\n' % self.meta['observations']
    string += '  reference frame: %s\n' % self.meta['reference_frame']
    string += '  longitude range: %.2f to %.2f\n' % (self.meta['min_longitude'], 
                                                    self.meta['max_longitude'])
    string += '  latitude range: %.2f to %.2f\n' % (self.meta['min_latitude'], 
                                                   self.meta['max_latitude'])
    string += '  lowest tolerance: %.4f\n' % self.meta['min_tolerance']
    return string


  #def write_data_array(self,output_file_name,times,zero_initial_value=True):
  def write_data_array(self,output_file_name,times):
    # find distance to next nearest time                         
    f = h5py.File(output_file_name,'w')
    names = self.keys()
    lon = [self[n].meta['longitude'] for n in names]
    lat = [self[n].meta['latitude'] for n in names]
    positions = np.array([lon,lat]).transpose()
    f['position'] = positions
    f['name'] = names
    f['time'] = times
    f.create_dataset('mean',shape=(len(times),len(names),3),dtype=float)
    f.create_dataset('mask',shape=(len(times),len(names)),dtype=bool)
    f.create_dataset('covariance',shape=(len(times),len(names),3,3),dtype=float)
    f.create_dataset('variance',shape=(len(times),len(names),3),dtype=float)
    for i,n in enumerate(names):
      logger.info('writing displacement data for station %s' % n)
      mean,cov = self[n](times)
      f['mean'][:,i,:] = mean.data
      f['mask'][:,i] = mean.mask[:,0]
      f['covariance'][:,i,:,:] = cov.data
      f['variance'][:,i,:] = cov.data[:,[0,1,2],[0,1,2]]

    f.close()

  def view(self,**kwargs):
    view_stationdb([self],**kwargs)
    
def view_stationdb(db_list,**kwargs):
  outfile_list = []
  buff_list = []
  for i,db in enumerate(db_list):
    dt = db.meta['min_tolerance']
    start = db.meta['start']
    end = db.meta['end']
    times = np.arange(start,end,dt)
    outfile = '.temp%s.h5' % i
    db.write_data_array(outfile,times)
    outfile_list += [outfile]
    buff_list += [h5py.File(outfile)]

  plot.view(buff_list,**kwargs)

  for buff in buff_list:
    buff.close()

  for outfile in outfile_list:
    os.remove(outfile)
  
  

    
    

  




