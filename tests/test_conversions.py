#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import gps.conversions
import scipy.linalg
import unittest
import logging
logging.basicConfig(level=logging.WARNING)

class Test(unittest.TestCase):
  def test_bound_lon_lat(self):
    test = (1.0,91.0)
    true = (-179.0,89.0)
    result = gps.conversions.bound_lon_lat(*test)
    self.assertTrue(true == result)

    test = (1.0,180.0)
    true = (-179.0,0.0)
    result = gps.conversions.bound_lon_lat(*test)
    self.assertTrue(true == result)

    test = (360.0,360.0)
    true = (0.0,0.0)
    result = gps.conversions.bound_lon_lat(*test)
    self.assertTrue(true == result)

    test = (1.0,-91.0)
    true = (-179.0,-89.0)
    result = gps.conversions.bound_lon_lat(*test)
    self.assertTrue(true == result)

    test = (-179.0,91.0)
    true = (1.0,89.0)
    result = gps.conversions.bound_lon_lat(*test)
    self.assertTrue(true == result)

    test = (361.0,91.0)
    true = (-179.0,89.0)
    result = gps.conversions.bound_lon_lat(*test)
    self.assertTrue(true == result)

    test = (-1.0,-89.0)
    true = (-1.0,-89.0)
    result = gps.conversions.bound_lon_lat(*test)
    self.assertTrue(true == result)

  def test_geodetic_to_ECEF(self):
    test = (0.0,0.0,0.0)
    true = (gps.conversions.ELLIPSOIDS['WGS84']['a'],0.0,0.0)
    result = gps.conversions.geodetic_to_ECEF(*test)            
    self.assertTrue(np.all(np.isclose(result,true)))

    test = (90.0,0.0,0.0)
    true = (0.0,gps.conversions.ELLIPSOIDS['WGS84']['a'],0.0)
    result = gps.conversions.geodetic_to_ECEF(*test)            
    self.assertTrue(np.all(np.isclose(result,true)))

    test = (180.0,0.0,0.0)
    true = (-gps.conversions.ELLIPSOIDS['WGS84']['a'],0.0,0.0)
    result = gps.conversions.geodetic_to_ECEF(*test)            
    self.assertTrue(np.all(np.isclose(result,true)))

  def test_ECEF_to_geodetic(self):
    true = (179.0,89.0,200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (-179.0,89.0,200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (179.0,-89.0,200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (-179.0,-89.0,200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (179.0,89.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (-179.0,89.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (179.0,-89.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (-179.0,-89.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (1.0,1.0,200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (-1.0,1.0,200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (1.0,-1.0,200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (-1.0,-1.0,200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (1.0,1.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (-1.0,1.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (1.0,-1.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

    true = (-1.0,-1.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    

  def test_ECEF_to_ENU(self):  
    true = (-1.0,-1.0,-200.0)
    ecef = gps.conversions.geodetic_to_ECEF(*true)            
    result = gps.conversions.ECEF_to_geodetic(*ecef)    
    self.assertTrue(np.all(np.isclose(result,true)))    


unittest.main()
