#!/usr/bin/env python
import numpy as np
import modest

# WGS84 spheroid parameters
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

mas_to_radians = np.pi/(1000.0*3600.0*180.0)

#units of m/year
T_VEL   = np.array([0.41, 0.22, 0.41])/1000.0

#units of m/year
T_SIGMA = np.array([0.27, 0.32, 0.30])/1000.0

# units of radians/year
ITRFPMM_VEL = {
  'AMUR': np.array([-0.190, -0.442,  0.915])*mas_to_radians,
  'ANTA': np.array([-0.252, -0.302,  0.643])*mas_to_radians,
  'ARAB': np.array([ 1.202, -0.054,  1.485])*mas_to_radians,
  'AUST': np.array([ 1.504,  1.172,  1.228])*mas_to_radians,
  'CARB': np.array([ 0.049, -1.088,  0.664])*mas_to_radians,
  'EURA': np.array([-0.083, -0.534,  0.750])*mas_to_radians,
  'INDI': np.array([ 1.232,  0.303,  1.540])*mas_to_radians,
  'NAZC': np.array([-0.330, -1.551,  1.625])*mas_to_radians,
  'NOAM': np.array([ 0.035, -0.662, -0.100])*mas_to_radians,
  'NUBI': np.array([ 0.095, -0.598,  0.723])*mas_to_radians,
  'PCFC': np.array([-0.411,  1.036, -2.166])*mas_to_radians,
  'SOAM': np.array([-0.243, -0.311, -0.154])*mas_to_radians,
  'SOMA': np.array([-0.080, -0.745,  0.897])*mas_to_radians,
  'SUND': np.array([ 0.047, -1.000,  0.975])*mas_to_radians
}

# units of radians/year
ITRFPMM_SIGMA = {
  'AMUR': np.array([0.040, 0.051, 0.049])*mas_to_radians,
  'ANTA': np.array([0.008, 0.006, 0.009])*mas_to_radians,
  'ARAB': np.array([0.082, 0.100, 0.063])*mas_to_radians,
  'AUST': np.array([0.007, 0.007, 0.007])*mas_to_radians,
  'CARB': np.array([0.201, 0.417, 0.146])*mas_to_radians,
  'EURA': np.array([0.008, 0.007, 0.008])*mas_to_radians,
  'INDI': np.array([0.031, 0.128, 0.030])*mas_to_radians,
  'NAZC': np.array([0.011, 0.029, 0.013])*mas_to_radians,
  'NOAM': np.array([0.008, 0.009, 0.008])*mas_to_radians,
  'NUBI': np.array([0.009, 0.007, 0.009])*mas_to_radians,
  'PCFC': np.array([0.007, 0.007, 0.009])*mas_to_radians,
  'SOAM': np.array([0.009, 0.010, 0.009])*mas_to_radians,
  'SOMA': np.array([0.028, 0.030, 0.012])*mas_to_radians,
  'SUND': np.array([0.381, 1.570, 0.045])*mas_to_radians
}

def bound_lon_lat(lon,lat):
  '''returns values of latitude and longitude which are equivalent to 
     the provided latitude and longitude but bounded such that 
     latitude is between -90 and 90, and longitude is between -180 
     and 180
  '''
  lat = (lat+90)%360 - 90
  if lon > 90.0:
    lon += 180.0
    lat  = (-lat[lat>90]+90)%180.0 - 90

  lon = (lon+180)%360 - 180
  return lon,lat
 

def geodetic_to_cartesian(lon,lat,h,ref='WGS84'):
  '''Converts geodetic coordinates to cartesians coordinates
  
  change to geodetic to ecef

  Parameters
  ----------
    lon: (scalar) longitude (degrees)
    lat: (scalar) latitude (degrees)
    h: (scalar) height above the ellipsoid (meters)

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
  out = geodetic_to_cartesian(lon,lat,h,ref)
  return np.array(out)


def cartesian_to_geodetic(x,y,z,ref='WGS84'):
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

  lon,lat = bound_lon_lat(lon,lat)
  return lon,lat,h


def cartesian_to_neu_matrix(lon,lat,h,ref='WGS84'):
  '''returns a 3 by 3 rotation matrix which takes a point in cartesian
  coordinates and maps to to a North, East, Up coordinate system
  
  change to ECEF to ENU

  Parameter
  ---------
    lon,lat,h: Longitude, Latitude, and height

  Returns
  -------
    out: (3,3) array which rotates a point in cartesian coordinates to
      a North, East, Up coordinate system

  '''
  a = ELLIPSOIDS[ref]['a']
  f = ELLIPSOIDS[ref]['f']
  b = a*(1 - f)

  lon = lon*np.pi/180
  lat = lat*np.pi/180

  xlat = (a*(1 - b**2/a**2)*np.sin(lat)*np.cos(lon)*np.cos(lat)**2
          /(-(1 - b**2/a**2)*np.sin(lat)**2 + 1)**(3/2) - 
          (a/np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1) + h)*
          np.sin(lat)*np.cos(lon))
  xlon = (-(a/np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1) + h)*
          np.sin(lon)*np.cos(lat))
  xh = np.cos(lon)*np.cos(lat)

  ylat = (a*(1 - b**2/a**2)*np.sin(lon)*np.sin(lat)*np.cos(lat)**2/
          (-(1 - b**2/a**2)*np.sin(lat)**2 + 1)**(3/2) - 
          (a/np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1) + h)*
          np.sin(lon)*np.sin(lat))
  ylon = ((a/np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1) + h)*
          np.cos(lon)*np.cos(lat))
  yh = np.sin(lon)*np.cos(lat)

  zlat = ((h + b**2/(a*np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1)))*
          np.cos(lat) + b**2*(1 - b**2/a**2)*np.sin(lat)**2*
          np.cos(lat)/(a*(-(1 - b**2/a**2)*
          np.sin(lat)**2 + 1)**(3/2)))
  zlon = 0*lat
  zh = np.sin(lat)

  north_length = np.sqrt(xlat**2 + ylat**2 + zlat**2)
  xlat,ylat,zlat = xlat/north_length,ylat/north_length,zlat/north_length

  east_length = np.sqrt(xlon**2 + ylon**2 + zlon**2)
  xlon,ylon,zlon = xlon/east_length,ylon/east_length,zlon/east_length

  up_length = np.sqrt(xh**2 + yh**2 + zh**2)
  xh,yh,zh = xh/up_length,yh/up_length,zh/up_length

  out = np.array([[xlat,ylat,zlat],
                  [xlon,ylon,zlon],
                  [xh,    yh,  zh]])

  return out


def cartesian_to_neu(cart_coor,geo_coor,ref='WGS84'):
  '''rotates vectors in cartesian coordinates to North, East, Up for
  given geodetic locations

  Change to ECEF to NEU

  Parameters
  ----------
    cart_coor: (N,3) array containing x, y, and z coordinates
    geo_coor: (N,3) array containing longitude, latitude, and height
      which specify the location on the reference ellispoid where
      North, East, and Up are defined.  
 
  Returns 
  -------
    out: (N,3) array of the vectors in cart_coor rotated to N, E, and
      U as defined by the corresponding geo_coor


  Note
  ----
    The results produces with Earth-like eccentricity are numerically 
    indistinguishable from the results produced when assuming a 
    spherical earth.  Only the north component is very slightly 
    changed.
  '''   
  a = ELLIPSOIDS[ref]['a']
  f = ELLIPSOIDS[ref]['f']
  b = a*(1 - f)

  lon = geo_coor[:,0]*np.pi/180
  lat = geo_coor[:,1]*np.pi/180
  h = geo_coor[:,2]*1.0

  xlat = (a*(1 - b**2/a**2)*np.sin(lat)*np.cos(lon)*np.cos(lat)**2
          /(-(1 - b**2/a**2)*np.sin(lat)**2 + 1)**(3/2) - 
          (a/np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1) + h)*
          np.sin(lat)*np.cos(lon))
  xlon = (-(a/np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1) + h)*
          np.sin(lon)*np.cos(lat))
  xh = np.cos(lon)*np.cos(lat)

  ylat = (a*(1 - b**2/a**2)*np.sin(lon)*np.sin(lat)*np.cos(lat)**2/
          (-(1 - b**2/a**2)*np.sin(lat)**2 + 1)**(3/2) - 
          (a/np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1) + h)*
          np.sin(lon)*np.sin(lat))
  ylon = ((a/np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1) + h)*
          np.cos(lon)*np.cos(lat))
  yh = np.sin(lon)*np.cos(lat)

  zlat = ((h + b**2/(a*np.sqrt(-(1 - b**2/a**2)*np.sin(lat)**2 + 1)))*
          np.cos(lat) + b**2*(1 - b**2/a**2)*np.sin(lat)**2*
          np.cos(lat)/(a*(-(1 - b**2/a**2)*
          np.sin(lat)**2 + 1)**(3/2)))
  zlon = 0*lat
  zh = np.sin(lat)

  north_length = np.sqrt(xlat**2 + ylat**2 + zlat**2)
  xlat,ylat,zlat = xlat/north_length,ylat/north_length,zlat/north_length

  east_length = np.sqrt(xlon**2 + ylon**2 + zlon**2)
  xlon,ylon,zlon = xlon/east_length,ylon/east_length,zlon/east_length

  up_length = np.sqrt(xh**2 + yh**2 + zh**2)
  xh,yh,zh = xh/up_length,yh/up_length,zh/up_length

  N = len(cart_coor)

  out = np.zeros((N,3))
  out[:,0] = xlat*cart_coor[:,0] + ylat*cart_coor[:,1] + zlat*cart_coor[:,2]
  out[:,1] = xlon*cart_coor[:,0] + ylon*cart_coor[:,1] + zlon*cart_coor[:,2]
  out[:,2] = xh*cart_coor[:,0] + yh*cart_coor[:,1] + zh*cart_coor[:,2]

  return out


def itrf2008pmm(lon,lat,h,plate):
  '''Returns the velocities at given sites for the given tectonic 
  plate based on the plate motion model from Altamimi et al 2012.
  Velocities given in m/year for the  North, East, and Up component

  Parameters
  ----------
    lon,lat,h: geodetic coordinates where velocities are to be output.
      These are the longitude, latidude, and height component
      plate: which tectonic plate angular velocity to use. This is a 
      string of one of the plate acronyms used by Altamimi et al 2012.

  Returns
  -------
    vel_neu: (3,) North, East, and Up component of velocities in m/year
    cov_neu: (3,3) covariance matrix of velocities

  '''
  x,y,z = geodetic_to_cartesian(lon,lat,h,ref='WGS84')
  vel_xyz = np.cross(ITRFPMM_VEL[plate],[x,y,z]) + T_VEL

  cov_xyz = [ITRFPMM_SIGMA[plate][1]**2 + ITRFPMM_SIGMA[plate][2]**2 + T_SIGMA[0]**2,
             ITRFPMM_SIGMA[plate][0]**2 + ITRFPMM_SIGMA[plate][2]**2 + T_SIGMA[1]**2,
             ITRFPMM_SIGMA[plate][0]**2 + ITRFPMM_SIGMA[plate][1]**2 + T_SIGMA[2]**2]
  cov_xyz = np.diag(cov_xyz)
  
  rotation_matrix = cartesian_to_neu_matrix(lon,lat,h,ref='WGS84')
  cov_neu = rotation_matrix.dot(cov_xyz).dot(rotation_matrix.transpose())
  vel_neu = rotation_matrix.dot(vel_xyz)
  return vel_neu,cov_neu

  
  









  
    
  
