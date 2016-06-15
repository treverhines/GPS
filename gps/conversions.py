#!/usr/bin/env python
from __future__ import division
import numpy as np
import modest

# This module is used to make conversions between geodetic coordinate
# systems, Earth-Centered, Earth-Fixed (ECEF) and local ENU coordinate
# systems


# Ellipsoid definitions in terms of their semi-major axis length and
# flattening
ELLIPSOIDS = {
  'NAD27':{
    'a':6378206.4,
    'f':1.0/294.978698214},
  'GRS80':{
    'a':6378137.0,
      'f':1.0/298.257222101},
  'WGS84':{
    'a':6378137.0,
      'f':1.0/298.257223563}}

def bound_lon_lat(lon,lat):
  '''Bounds longitude and latitude to be between [-180,180] and [-90,90]
  respectively. 

  PARAMETERS 
  ----------

    lon: longitude in degrees
    lat: latitude in degrees

  Returns 
  ------- 

    bounded_lon, bounded_lat

  '''
  # first restrict latitude to be between -180 and 180
  lat = (lat+180)%360 - 180

  # if latitude is between 90 and 270 then subtract 180 and flip the sign
  # longitude also then needs 180 added to it
  if (lat >= 90):
    lat -= 180
    lat *= -1
    lon += 180

  if (lat < -90):
    lat += 180
    lat *= -1
    lon += 180

  # bound longitude to be between -180 and 180
  lon = (lon + 180)%360 - 180
  return lon,lat
 

def geodetic_to_ECEF(lon,lat,h,ref='WGS84'):
  '''Converts geodetic coordinates to cartesians coordinates
  
  change to geodetic to ecef

  Parameters
  ----------
    lon: longitude (degrees)
    lat: latitude (degrees)
    h: height above the ellipsoid (meters)
    ref: (optional) reference ellipoid. Either 'NAD27','GRS80', or 
      'WGS84'

  Returns
  -------
    x,y,z coordinates in meters

  The coordinate system is as defined by the WGS84, where the z axis
  is pointing in the direction of the IERS Reference pole; the x axis
  is the intersection between the equatorial plane and the IERS
  Reference Meridian; and the y axis completes a right-handed
  coordinate coordinate system
  '''
  lon = lon*np.pi/180
  lat = lat*np.pi/180

  # set ellipsoid parameters
  a = ELLIPSOIDS[ref]['a']
  f = ELLIPSOIDS[ref]['f']
  c = a*(1 - f)
  e2 = 1 - (c/a)**2
  n = a/np.sqrt(1 - e2*np.sin(lat)**2)

  x = (n+h)*np.cos(lon)*np.cos(lat)
  y = (n+h)*np.sin(lon)*np.cos(lat)
  z = (n*(c/a)**2+h)*np.sin(lat)

  return x,y,z


def _system(coor,ref):
  '''used as the forward problem when solving for the 
     geodetic coordinates
  '''
  lon = coor[0]
  lat = coor[1]
  h = coor[2]
  out = geodetic_to_ECEF(lon,lat,h,ref)
  return np.array(out)


def ECEF_to_geodetic(x,y,z,ref='WGS84'):
  '''finds the geodetic coordinates from cartesian coordinates

  This is done by solving a nonlinear inverse problem using 
  'geodetic_to_cartesian' as the forward problem. This is not the
  most efficient algorithm.

  change to ecef to geodetic

  Parameters
  ----------
    x,y,z: (scalars) coordinates in meters

    ref: (optional) reference ellipoid. Either 'NAD27','GRS80', or 
      'WGS84'

  Returns
  -------
    lon,lat,h: longitude (degrees), latitude (degrees) and height 
      above the reference ellipsoid (meters)

  The coordinate system is as defined by the WGS84, where the z axis
  is pointing in the direction of the IERS Reference pole; the x axis
  is the intersection between the equatorial plane and the IERS
  Reference Meridian; and the y axis completes a right-handed
  coordinate coordinate system
  '''
  lon,lat,h =  modest.nonlin_lstsq(
                 _system,
                 np.array([x,y,z]),
                 np.zeros(3),
                 system_args=(ref,),
                 rtol=1e-10,
                 atol=1e-10,
                 maxitr=100,
                 LM_damping=True,
                 LM_param=1.0)

  #print(lon,lat)
  lon,lat = bound_lon_lat(lon,lat)
  return lon,lat,h


def ENU_to_ECEF_matrix(lon,lat,h):
  lon *= np.pi/180
  lat *= np.pi/180
  out = np.array([[-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
                  [ np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
                  [0.0,                       np.cos(lat),             np.sin(lat)]])
  return out


def ECEF_to_ENU_matrix(lon,lat,h):
  out = ENU_to_ECEF_matrix(lon,lat,h).transpose()
  return out


def ENU_to_ECEF(de,dn,du,lon,lat,h):
  ''' 
  Takes a change in east, north, and up directions and returns a change
  in x, y, and z directions
  '''
  enu = np.array([de,dn,du])
  ecef = ENU_to_ECEF_matrix(lon,lat,h).dot(enu)
  return ecef


def ECEF_to_ENU(dx,dy,dz,lon,lat,h):
  ''' 
  Takes a change in x, y, and z directions and returns a change
  in east, north, and up directions
  '''
  ecef = np.array([dx,dy,dz])
  enu = ECEF_to_ENU_matrix(lon,lat,h).dot(ecef)
  return enu


  
  









  
    
  
