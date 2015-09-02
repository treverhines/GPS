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
  metadata = {'name':None,
              'id':None,
              'start':None,
              'end':None,
              'reference frame':None,
              'longitude':None,
              'latitude':None,
              'height':None,
              'observations':None}

  data = {'time':None,
          'east disp.':None,
          'north disp.':None,
          'vert. disp.':None,
          'east std. dev.':None,
          'north std. dev.':None,
          'vert. std. dev.':None,
          'north-east corr.':None,
          'north-vert. corr.':None,
          'east-vert. corr.':None}

  version = find_field('Format Version',string)
  if version == '1.1.0':
    metadata['reference'] = find_field('reference frame',string)
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
    data['time'] = np.array(time)
    metadata['observations'] = len(data['time'])

    data['north disp.'] = np.array(data_values[:,DN_idx],dtype=float)
    data['east disp.'] = np.array(data_values[:,DE_idx],dtype=float)
    data['vert. disp.'] = np.array(data_values[:,DU_idx],dtype=float)

    data['north std. dev.'] = np.array(data_values[:,Sn_idx],dtype=float)
    data['east std. dev.'] = np.array(data_values[:,Se_idx],dtype=float)
    data['vert. std. dev.'] = np.array(data_values[:,Su_idx],dtype=float)

    data['north-east corr.'] = np.array(data_values[:,Rne_idx],dtype=float)
    data['north-vert. corr.'] = np.array(data_values[:,Rnu_idx],dtype=float)
    data['east-vert. corr.'] = np.array(data_values[:,Reu_idx],dtype=float)

  else:
    logger.warning('version not recognized')

  return metadata,data

class Station:
  def __init__(self,pos_file_name,tol=1/365):
    f = open(pos_file_name,'r')
    pos_string = f.read()
    f.close()
    meta,data = parse_pos_string(pos_string)
    meta['interp. tolerance'] = tol

    time = data['time']

    val = np.array([data['east disp.'],
                    data['north disp.'],
                    data['vert. disp.']])

    val = np.einsum('ij->ji',val)

    cov = [[data['east std. dev.']**2,
            data['north-east corr.']*data['east std. dev.']*data['north std. dev.'],
            data['east-vert. corr.']*data['east std. dev.']*data['vert. std. dev.']],
           [data['north-east corr.']*data['east std. dev.']*data['north std. dev.'],
            data['north std. dev.']**2,
            data['north-vert. corr.']*data['north std. dev.']*data['vert. std. dev.']],
           [data['east-vert. corr.']*data['east std. dev.']*data['vert. std. dev.'],
            data['north-vert. corr.']*data['north std. dev.']*data['vert. std. dev.'],
            data['vert. std. dev.']**2]]

    cov = np.einsum('ijk->kij',cov)
    self.meta = meta 
    self.time = time
    self.val = val
    self.cov = cov
    self.val_itp = maskitp.MaskedNearestInterp(time,val,tol)
    self.cov_itp = maskitp.MaskedNearestInterp(time,cov,tol)

  def __repr__(self):
    string = '\nStation\n'
    string += '  id: %s (%s)\n' % (self.meta['id'],self.meta['name']) 
    string += '  time range: %.2f to %.2f\n' % (self.meta['start'],self.meta['end']) 
    string += '  observations: %s\n' % self.meta['observations']
    string += '  reference: %s\n' % self.meta['reference']
    string += '  longitude: %.2f\n' % self.meta['longitude']
    string += '  latitude: %.2f\n' % self.meta['latitude']
    string += '  height: %.2f m\n' % self.meta['height']
    string += '  interp. tolerance: %.4f years (%.1f days)' % (
                 self.meta['interp. tolerance'],
                 self.meta['interp. tolerance']*365)
    return string

  def __call__(self,t):
    return self.val_itp(t),self.cov_itp(t)        

  def __getitem__(self,i):
    return self.val[i],self.cov[i]        


class StationDB(collections.OrderedDict):
  def __init__(self,database_directory,tol=1/365):
    collections.OrderedDict.__init__(self)
    pos_files = os.listdir(database_directory)
    # sort stations alphabetically
    pos_files = np.sort(pos_files)
    pos_files = [database_directory+'/'+i for i in pos_files]
    self.meta = {'start':np.inf,
                 'end':-np.inf,
                 'reference':None,
                 'stations':0,
                 'observations':0,
                 'min_longitude':180,
                 'max_longitude':-180,
                 'min_latitude':90,
                 'max_latitude':-90}
    for i in pos_files:
      sta = Station(i,tol=tol)
      logger.info('initializing station %s:%s' % (sta.meta['id'],sta.__repr__()))
      if self.meta['reference'] is None:
        self.meta['reference'] = sta.meta['reference']

      if self.meta['reference'] != sta.meta['reference']:
        logger.warning(
          'reference frame for station %s, %s, is not the same as '
          'previously added stations' % (sta['id'],sta['reference']))

      if self.meta['start'] > sta.meta['start']:
        self.meta['start'] = sta.meta['start']

      if self.meta['end'] < sta.meta['end']:
        self.meta['end'] = sta.meta['end']

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

      self[sta.meta['id']] = sta
        

  def __repr__(self):
    string = '\nStationDB\n'
    string += '  time range: %.2f to %.2f\n' % (self.meta['start'],self.meta['end']) 
    string += '  stations: %s\n' % self.meta['stations']
    string += '  observations: %s\n' % self.meta['observations']
    string += '  reference: %s\n' % self.meta['reference']
    string += '  longitude range: %.2f to %.2f\n' % (self.meta['min_longitude'], 
                                                    self.meta['max_longitude'])
    string += '  latitude range: %.2f - %.2f\n' % (self.meta['min_latitude'], 
                                                   self.meta['max_latitude'])
    return string


  def write_data_array(self,output_file_name,times,zero_initial_value=True):
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
    for i,n in enumerate(names):
      logger.info('writing displacement data for station %s' % n)
      mean,cov = self[n](times)
      if zero_initial_value:
        unmasked_rows = np.nonzero(~mean.mask)[0]
        if len(unmasked_rows) != 0:
          initial_unmasked_row = unmasked_rows[0]
          initial_unmasked_disp = mean[initial_unmasked_row,:] 
          mean -= initial_unmasked_disp

      f['mean'][:,i,:] = mean.data
      f['mask'][:,i] = mean.mask[:,0]
      f['covariance'][:,i,:,:] = cov.data

    f.close()



    
    

  




