#!/usr/bin/env python
from __future__ import division
import numpy as np
import conversions

# This modules calculates plate motions based on the results 
# of Altamimi et al 2012

# milliarcseconds to radians
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

def itrf2008pmm(lon,lat,h,plate):
  '''
  Returns the velocities at given sites for the given tectonic 
  plate based on the plate motion model from Altamimi et al 2012.
  Velocities given in m/year for the  East, North and Up component

  Parameters
  ----------
    lon,lat,h: geodetic coordinates where velocities are to be output.
      These are the longitude, latidude, and height component

    plate: which tectonic plate angular velocity to use. This is a 
      string of one of the plate acronyms used by Altamimi et al 2012.

  Returns
  -------
    vel_enu: (3,) North, East, and Up component of velocities in m/year
    cov_enu: (3,3) covariance matrix of velocities

  '''
  x,y,z = conversions.geodetic_to_ECEF(lon,lat,h,ref='WGS84')
  vel_xyz = np.cross(ITRFPMM_VEL[plate],[x,y,z]) + T_VEL

  cov_xyz = [ITRFPMM_SIGMA[plate][1]**2 + ITRFPMM_SIGMA[plate][2]**2 + T_SIGMA[0]**2,
             ITRFPMM_SIGMA[plate][0]**2 + ITRFPMM_SIGMA[plate][2]**2 + T_SIGMA[1]**2,
             ITRFPMM_SIGMA[plate][0]**2 + ITRFPMM_SIGMA[plate][1]**2 + T_SIGMA[2]**2]
  cov_xyz = np.diag(cov_xyz)
  
  R = ECEF_to_ENU_matrix(lon,lat,h,ref='WGS84')
  cov_enu = R.dot(cov_xyz).dot(R.transpose())
  vel_enu = R.dot(vel_xyz)
  return vel_enu,cov_enu

  
  









  
    
  
