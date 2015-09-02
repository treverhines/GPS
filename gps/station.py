#!/usr/bin/env python
from __future__ import division
import numpy as np
import datetime
import time 
import re
import logging
import conversions
import modest
import maskitp
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
  metadata = {'station name':None,
              'station ID':None,
              'start time':None,
              'end time':None,
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
    metadata['reference frame'] = find_field('reference frame',string)
    metadata['station name'] = find_field('station name',string)
    metadata['station ID'] = find_field('4-character ID',string)
    start_time_string = find_field('First Epoch',string)
    metadata['start time'] = string_to_decyear(start_time_string,'%Y%m%d %H%M%S')
    end_time_string = find_field('Last Epoch',string)
    metadata['end time'] = string_to_decyear(end_time_string,'%Y%m%d %H%M%S')
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
    val = np.array([data['east disp.'],
                    data['north disp.'],
                    data['vert. disp.']])
    val = np.einsum('ij->ji',val)

    cov = [[data['east std. dev.']**2,
            data['north-east corr.']*(data['east std. dev.']*data['north std. dev.']),
            data['east-vert. corr.']*(data['east std. dev.']*data['vert. std. dev.'])],
           [data['north-east corr.']*(data['east std. dev.']*data['north std. dev.']),
            data['north std. dev.']**2,
            data['north-vert. corr.']*(data['north std. dev.']*data['vert. std. dev.'])],
           [data['east-vert. corr.']*(data['east std. dev.']*data['vert. std. dev.']),
            data['north-vert. corr.']*(data['north std. dev.']*data['vert. std. dev.']),
            data['vert. std. dev.']**2]]

    cov = np.einsum('ijk->kij',cov)
    self.meta = meta 
    self.val = val
    self.cov = cov
    self.val_itp = maskitp.MaskedNearestInterp(data['time'],val,tol)
    self.cov_itp = maskitp.MaskedNearestInterp(data['time'],cov,tol)

  def __repr__(self):
    string = 'Station\n'
    string += '  id: %s (%s)\n' % (self.meta['station ID'],self.meta['station name']) 
    string += '  time range: %s - %s\n' % (self.meta['start time'],self.meta['end time']) 
    string += '  observations: %s\n' % self.meta['observations']
    string += '  reference frame: %s\n' % self.meta['reference frame']
    string += '  longitude: %s\n' % self.meta['longitude']
    string += '  latitude: %s\n' % self.meta['latitude']
    string += '  height: %s' % self.meta['height']
    return string

  def __call__(self,t):
    return self.val_itp(t),self.cov_itp(t)        

  def __getitem__(self,i):
    return self.val[i],self.cov[i]        

class StationDB:
  pass
    
    

  




