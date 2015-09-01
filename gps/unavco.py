#!/usr/bin/env python
import os
import urllib
import time  as timemod
import subprocess as sp
import numpy as np
import dateutil.parser
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse
from matplotlib.widgets import Slider
import misc
import pandas
import logging
from tplot.basemap import Basemap
from tplot.quiver import Quiver
matplotlib.quiver.Quiver = Quiver # for error ellipses

logger = logging.getLogger(__name__)

TODAY = misc.decyear(*tuple(timemod.gmtime())[:3]) #today in decimal year (eg. 2014.125)

try:
  GPSPATH = os.environ['GPSPATH']
except KeyError:
  print('\ncannot find UNAVOPATH:\n\n'
        'make an empty directory to store data downloaded from\n' 
        'unavco.org and save that directory location in your\n' 
        'environment as "UNAVCOPATH"\n')
  

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
                '<UNAVCOPATH>/<repository>'

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
  print(file_string)
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




