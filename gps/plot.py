#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.widgets import Slider
from myplot.basemap import Basemap
from myplot.quiver import Quiver
import misc
import matplotlib
matplotlib.quiver.Quiver = Quiver # for error ellipses

def quiver_args(position,disp_array,cov_array=None,mask=None):
  N = len(position)
  if mask is None:
    mask = np.zeros(N,dtype=bool)

  x = position[:,0]
  y = position[:,1]
  u = disp_array[:,0]
  u[mask] = 0.0
  v = disp_array[:,1]
  v[mask] = 0.0

  if cov_array is not None:
    var_u = cov_array[:,0,0]
    var_v = cov_array[:,1,1]
    cov_uv = cov_array[:,0,1]
    #var_u = cov_array[:,0]
    #var_v = cov_array[:,1]
    #cov_uv = 0.0*var_u
    var_u[mask] = 1e-8
    var_v[mask] = 1e-8
    cov_uv[mask] = 1e-8
    sigma_u = np.sqrt(var_u)
    sigma_v = np.sqrt(var_v)
    rho = cov_uv/(sigma_u*sigma_v)
    return (x,y,u,v,(sigma_u,sigma_v,rho))

  else:
    return (x,y,u,v)


def create_default_basemap(lat_lst,lon_lst):
  '''
  creates a basemap that bounds lat_lst and lon_lst
  '''
  lon_buff = (max(lon_lst) - min(lon_lst))/10.0
  lat_buff = (max(lat_lst) - min(lat_lst))/10.0
  if lon_buff < 0.5:
    lon_buff = 0.5

  if lat_buff < 0.5:
    lat_buff = 0.5

  llcrnrlon = min(lon_lst) - lon_buff
  llcrnrlat = min(lat_lst) - lat_buff
  urcrnrlon = max(lon_lst) + lon_buff
  urcrnrlat = max(lat_lst) + lat_buff
  lon_0 = (llcrnrlon + urcrnrlon)/2.0
  lat_0 = (llcrnrlat + urcrnrlat)/2.0
  return Basemap(projection='tmerc',
                 lon_0 = lon_0,
                 lat_0 = lat_0,
                 llcrnrlon = llcrnrlon,
                 llcrnrlat = llcrnrlat,
                 urcrnrlon = urcrnrlon,
                 urcrnrlat = urcrnrlat,
                 resolution = 'h') 


def view(data_list,
         name_list=None,
         units=None,
         draw_map=True,
         quiver_scale=0.00001,
         scale_length=1.0):
  # data list is a list of dictionary-like objects with keys: mean,
  # covariance, mask, position, time
  mask_list = []
  mean_list = []
  cov_list = []
  if name_list is None:
    name_list = ['displacement %s' % i for i in range(len(data_list))]

  for data in data_list:
    mask_list += [data['mask']]
    mean_list += [data['mean']]
    cov_list += [data['covariance']]
    
  times = data_list[0]['time'][:]
  lon = data_list[0]['position'][:,0]
  lat = data_list[0]['position'][:,1]
  station_names = data_list[0]['name'][:]

  _view(mean_list,
        cov_list,
        times,
        station_names,
        lon,
        lat,
        mask_list,
        units=units,
        disp_type=name_list,
        draw_map=draw_map,
        quiver_scale=quiver_scale,
        scale_length=scale_length)

def _view(displacement_list,
          covariance_list,
          times,
          station_names,
          lon,
          lat,
          mask,
          units='meters',
          disp_type=None,
          colors=None,
          draw_map=False,
          scale_length=1.0,
          quiver_scale=0.0001,
          map_resolution=200,
          artists=None):

  lonlat = np.array([lon,lat]).T
  #times -= 2010.0
  N = len(displacement_list)
  Nx = len(lon)
  if disp_type is None:
    disp_type = ['']*N
  
  if colors is None:
    colors = ['k','b','r','g','m']

  if artists is None:
    artists = []

  # setup background of main figure
  sub_fig = plt.figure('Time Series',figsize=(9.0,6.6))
  main_fig = plt.figure('Map View',figsize=(10,11.78))

  slider_ax = main_fig.add_axes([0.08,0.88,0.76,0.04])
  sub_ax1 = sub_fig.add_subplot(311)
  sub_ax2 = sub_fig.add_subplot(312)
  sub_ax3 = sub_fig.add_subplot(313)
  main_ax = main_fig.add_axes([0.08,0.08,0.76,0.76])

  time_slider = Slider(slider_ax,'time',
                       min(times),max(times),
                       valinit=min(times),
                       color='black')
  time = min(times)
  time_idx = np.argmin(abs(times - time))

  if draw_map is True:
    basemap = create_default_basemap(lat,lon)
    position = basemap(lon,lat)
    position = np.array(position).transpose()
    main_ax.patch.set_facecolor([0.0,0.0,1.0,0.2])
    basemap.drawtopography(ax=main_ax,vmin=-6000,vmax=4000,
                           alpha=0.6,resolution=map_resolution,zorder=0)
    basemap.drawcoastlines(ax=main_ax,linewidth=1.5,zorder=1)
    basemap.drawcountries(ax=main_ax,linewidth=1.5,zorder=1)
    basemap.drawstates(ax=main_ax,linewidth=1,zorder=1)
    basemap.drawrivers(ax=main_ax,linewidth=1,zorder=1)
    basemap.drawmeridians(np.arange(np.floor(basemap.llcrnrlon),
                        np.ceil(basemap.urcrnrlon),1.0),
                        labels=[0,0,0,1],dashes=[2,2],
                        ax=main_ax,zorder=1)
    basemap.drawparallels(np.arange(np.floor(basemap.llcrnrlat),
                        np.ceil(basemap.urcrnrlat),1.0),
                        labels=[1,0,0,0],dashes=[2,2],
                        ax=main_ax,zorder=1)
    basemap.drawmapscale(units='km',
                       lat=basemap.latmin+(basemap.latmax-basemap.latmin)/10.0,
                       lon=basemap.lonmax-(basemap.lonmax-basemap.lonmin)/5.0,
                       fontsize=16,
                       lon0=(basemap.lonmin+basemap.lonmax)/2.0,
                       lat0=(basemap.latmin+basemap.latmax)/2.0,
                       barstyle='fancy',ax=main_ax,
                       length=100,zorder=10)
    vertical_image = [basemap.drawscalar(0*position[:,0],lonlat,cmap=matplotlib.cm.seismic,zorder=-1,vmin=-0.01,vmax=0.01)]
    plt.colorbar(vertical_image[0])
  else:
    position = np.array([lon,lat]).transpose()
    main_ax.set_aspect('equal')

  station_point_lst = []
  station_point_label_lst = []
  for sid in range(Nx):
    loni = lon[sid]
    lati = lat[sid]
    x,y = position[sid,:]
    #x,y = basemap(loni,lati)
    station_point = main_ax.plot(x,y,'ko',markersize=3,picker=8,zorder=2)
    station_point_label_lst += [station_point[0].get_label()]
    station_point_lst += station_point

  station_point_label_lst = np.array(station_point_label_lst,dtype=str)

  Q_lst = []
  for idx in range(N):
    if covariance_list[idx] is not None:
      args = quiver_args(position,
                         displacement_list[idx][time_idx,:,:],
                         covariance_list[idx][time_idx,:,:,:],
                         mask[idx][time_idx,:])

      Q_lst += [main_ax.quiver(args[0],args[1],args[2],args[3],sigma=args[4],
                               scale_units='xy',
                               angles='xy',
                               width=0.004,
                               scale=quiver_scale,
                               color=colors[idx],
                               ellipse_edgecolors=colors[idx],
                               zorder=3)]
    else:
      args = quiver_args(position,
                         displacement_list[idx][time_idx,:,:])

      Q_lst += [main_ax.quiver(args[0],args[1],args[2],args[3],
                               scale_units='xy',
                               angles='xy',
                               width=0.004,
                               scale=quiver_scale,
                               color=colors[idx],
                               ellipse_edgecolors=colors[idx],
                               zorder=3)]

  u_scale = np.array([scale_length])
  v_scale = np.array([0.0])
  z_scale = np.array([0.0])
  xlim = main_ax.get_xlim()
  ylim = main_ax.get_ylim()
  xo = xlim[0]
  yo = ylim[0]
  xwidth = xlim[1] - xlim[0]
  ywidth = ylim[1] - ylim[0]
  x_scale = np.array([xo + xwidth/10.0])
  y_scale = np.array([yo + ywidth/10.0])
  x_text = xo + xwidth/10.0
  y_text = yo + ywidth/15.0
  dy_text = ywidth/20.0

  for i in range(N):
    main_ax.text(x_text,y_text+i*dy_text,disp_type[i],fontsize=16)
    main_ax.quiver(x_scale,y_scale+i*dy_text,u_scale,v_scale,
                  scale_units='xy',
                  angles='xy',
                  width=0.004,
                  scale=quiver_scale,
                  color=colors[i])

  main_ax.text(x_text,y_text+N*dy_text,
               '%s %s' % (np.round(scale_length,3),units),
               fontsize=16)

  def _slider_update(t):
    time_idx = np.argmin(abs(t - times))
    vertical_image[0].remove()
    vert_mask = mask[0][time_idx,:]
    vert_disp = displacement_list[0][time_idx,~vert_mask,2]
    vertical_image[0] = basemap.drawscalar(vert_disp,lonlat[~vert_mask],
                                           cmap=matplotlib.cm.seismic,
                                           zorder=-1,vmin=-0.01,vmax=0.01)
    #plt.colorbar(vertical_image[0])
    for idx in range(N):
      if covariance_list[idx] is not None:      
        args = quiver_args(position,
                           displacement_list[idx][time_idx,:,:],
                           covariance_list[idx][time_idx,:,:,:],
                           mask[idx][time_idx,:])
    
        Q_lst[idx].set_UVC(args[2],args[3],sigma=args[4])

      else: 
        args = quiver_args(position,
                           displacement_list[idx][time_idx])
    
        Q_lst[idx].set_UVC(args[2],args[3])

    main_fig.canvas.draw()
    return

  def _onpick(event):
    idx, = np.nonzero(str(event.artist.get_label()) == station_point_label_lst)
    station_label = station_names[idx[0]]
    sub_ax1.cla()
    sub_ax2.cla()
    sub_ax3.cla()
    for i in range(N):
      midx = mask[i][:,idx]  
      disp = displacement_list[i][:,idx,:]
      cov = covariance_list[i][:,idx,:,:]
      if covariance_list[i] is not None:      
        sub_ax1.errorbar(times[~midx],
                         disp[~midx,0], 
                         np.sqrt(cov[~midx,0,0]),
                         color=colors[i],capsize=0,fmt='.')
        sub_ax2.errorbar(times[~midx],
                         disp[~midx,1], 
                         np.sqrt(cov[~midx,1,1]),
                         color=colors[i],capsize=0,fmt='.')
        sub_ax3.errorbar(times[~midx],
                         disp[~midx,2], 
                         np.sqrt(cov[~midx,2,2]),
                         color=colors[i],capsize=0,fmt='.')
      else:
        sub_ax1.plot(times[~midx],
                     disp[~midx,0], 
                     color=colors[i])
        sub_ax2.plot(times[~midx],
                     disp[~midx,1], 
                     color=colors[i])
        sub_ax3.plot(times[~midx],
                     disp[~midx,2],
                     color=colors[i])

    sub_ax1.set_title(station_label,fontsize=16)
    sub_ax1.set_ylabel('easting',fontsize=16)
    sub_ax2.set_ylabel('northing',fontsize=16)
    sub_ax3.set_ylabel('vertical',fontsize=16)
    sub_fig.canvas.draw()
    event.artist.set_markersize(10)
    main_fig.canvas.draw()
    event.artist.set_markersize(3.0)
    return

  time_slider.on_changed(_slider_update)
  main_fig.canvas.mpl_connect('pick_event',_onpick)

  for a in artists:
    ax.add_artist(a)  

  plt.show()




