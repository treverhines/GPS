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
import modest
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
    var_u[mask] = 1e-10
    var_v[mask] = 1e-10
    cov_uv[mask] = 1e-10
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
  lon_buff = (max(lon_lst) - min(lon_lst))/20.0
  lat_buff = (max(lat_lst) - min(lat_lst))/20.0
  if lon_buff < 0.2:
    lon_buff = 0.2

  if lat_buff < 0.2:
    lat_buff = 0.2

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

def record_section(data_list):
  return


def view(data_list,
         name_list=None,
         units=None,
         draw_map=True,
         basemap=None, 
         title='displacement at %s',
         quiver_scale=0.00001,
         quiverscale_lonlat=None,
         mapscale_lonlat=None,
         minimap_pos=None,  
         vmin=-20,vmax=20,
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
        title=title,
        basemap=basemap, 
        disp_type=name_list,
        draw_map=draw_map,
        vmin=vmin,vmax=vmax,
        mapscale_lonlat=mapscale_lonlat,
        quiverscale_lonlat=quiverscale_lonlat,
        minimap_pos=minimap_pos,
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
          title='displacement at %s',
          disp_type=None,
          basemap=None, 
          colors=None,
          draw_map=False,
          scale_length=1.0,
          quiver_scale=0.0001,
          map_resolution=200,
          mapscale_lonlat = None, 
          quiverscale_lonlat = None, 
          minimap_pos = None, 
          vmin=-20,vmax=20,
          artists=None):

  lonlat = np.array([lon,lat]).T
  #times -= 2010.0
  N = len(displacement_list)
  Nx = len(lon)
  if disp_type is None:
    disp_type = ['']*N
  
  disp_type = [i.replace('_',' ') for i in disp_type]
  if colors is None:
    colors = ['k','g','r','b','m']

  if artists is None:
    artists = []

  sub_fig,sub_ax = plt.subplots(3,1,figsize=(8,6),sharex=True)
  # do not use scientific notation
  sub_ax[0].ticklabel_format(useOffset=False, style='plain')

  slider_fig = plt.figure('time slider',figsize=(8,1))
  slider_ax = slider_fig.add_axes([0.1,0.1,0.8,0.8])
  time_slider = Slider(slider_ax,'time',
                       min(times),max(times),
                       valinit=min(times),
                       color='black')

  main_fig = plt.figure('Map View',figsize=(12,6))
  main_ax = main_fig.add_axes([0.1,0.1,0.8,0.8])
  
  time = min(times)

  try:
    # if there is a %s in the title string then try to fill it with a timestamp
    main_ax.set_title(title % modest.decyear_inv(time,format='%Y-%m-%d'))
  except TypeError:
    main_ax.set_title(title)
  
  time_idx = np.argmin(abs(times - time))

  if draw_map is True:
    # figure z ordering
    # 0 topography
    # 1 geographic lines
    # 2 vertical displacement 
    # 3 displacement vectors
    # 4 legends
    if basemap is None:
      basemap = create_default_basemap(lat,lon)
    position = basemap(lon,lat)
    position = np.array(position).transpose()

    main_ax.patch.set_facecolor([1.0,1.0,1.0,0.0])

    # zorder 0
    basemap.drawtopography(ax=main_ax,vmin=-6000,vmax=4000,
                           alpha=0.4,resolution=map_resolution,zorder=0)

    # zorder 1
    basemap.drawcoastlines(ax=main_ax,linewidth=2.0,zorder=1,color=(0.5,0.5,0.5,1.0))
    basemap.drawcountries(ax=main_ax,linewidth=2.0,zorder=1,color=(0.5,0.5,0.5,1.0))
    basemap.drawstates(ax=main_ax,linewidth=2.0,zorder=1,color=(0.5,0.5,0.5,1.0))
    basemap.drawmeridians(np.arange(np.floor(basemap.llcrnrlon),
                          np.ceil(basemap.urcrnrlon),1.0),
                          labels=[0,0,0,1],dashes=[2,2],
                          ax=main_ax,zorder=1,color=(0.5,0.5,0.5,1.0))
    basemap.drawparallels(np.arange(np.floor(basemap.llcrnrlat),
                          np.ceil(basemap.urcrnrlat),1.0),
                          labels=[1,0,0,0],dashes=[2,2],
                          ax=main_ax,zorder=1,color=(0.5,0.5,0.5,1.0))

    # zorder 2
    vertical_image = [basemap.drawscalar(0*position[:,0],lonlat,
                                         cmap=matplotlib.cm.seismic,
                                         zorder=2,vmin=vmin,vmax=vmax,ax=main_ax,alpha=0.5)]

    # zorder 4
    if mapscale_lonlat is None:
      basemap.drawmapscale(units='km',
                         lat=basemap.latmin+(basemap.latmax-basemap.latmin)/3.25,
                         lon=basemap.lonmax-(basemap.lonmax-basemap.lonmin)/7.75,
                         fontsize=12,
                         lon0=(basemap.lonmin+basemap.lonmax)/2.0,
                         lat0=(basemap.latmin+basemap.latmax)/2.0,
                         barstyle='fancy',ax=main_ax,
                         length=100,zorder=4)
    else:
      basemap.drawmapscale(units='km',
                           lat=mapscale_lonlat[1],
                           lon=mapscale_lonlat[0],
                           fontsize=12,
                           lon0=(basemap.lonmin+basemap.lonmax)/2.0,
                           lat0=(basemap.latmin+basemap.latmax)/2.0,
                           barstyle='fancy',ax=main_ax,
                           length=100,zorder=4)


    cbar = plt.colorbar(vertical_image[0])
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.solids.set_rasterized(True)
    cbar.ax.set_ylabel('vertical displacement (mm)')
    if minimap_pos is None:
      basemap.drawminimap([0.587,0.7,0.15,0.15],ax=main_ax)
    else:
      basemap.drawminimap(minimap_pos,ax=main_ax)
    #basemap.drawminimap([0.095,0.72,0.15,0.15],ax=main_ax)
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
    station_point = main_ax.plot(x,y,'ko',markersize=3,picker=8,zorder=3)
    station_point_label_lst += [station_point[0].get_label()]
    station_point_lst += station_point

  station_point_label_lst = np.array(station_point_label_lst,dtype=str)

  Q_lst = []
  for idx in range(N):
    if np.any(covariance_list[idx] > 1e-8):
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
                               zorder=3+idx)]
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
                               zorder=3+idx)]

  u_scale = np.array([scale_length])
  v_scale = np.array([0.0])
  z_scale = np.array([0.0])
  xlim = main_ax.get_xlim()
  ylim = main_ax.get_ylim()
  xo = xlim[1]
  yo = ylim[1]
  xwidth = xlim[1] - xlim[0]
  ywidth = ylim[1] - ylim[0]
  #                       lat=basemap.latmax-(basemap.latmax-basemap.latmin)/3.25,
  #                       lon=basemap.lonmax-(basemap.lonmax-basemap.lonmin)/7.75,
  if quiverscale_lonlat is None:
    x_scale = np.array([xo - 2.25*xwidth/10.0])
    y_scale = np.array([yo - 4.4*ywidth/10.0])
  else:
    x_scale,y_scale = basemap(*quiverscale_lonlat)

  x_text = x_scale
  y_text = y_scale - 0.5*ywidth/10.0 
  dy_text = -ywidth/10.0

  for i in range(N):
    main_ax.text(x_text,y_text+i*dy_text,disp_type[i])
    main_ax.quiver(x_scale,y_scale+i*dy_text,u_scale,v_scale,
                   scale_units='xy',
                   angles='xy',
                   width=0.004,
                   scale=quiver_scale,
                   color=colors[i],zorder=4)

  main_ax.text(x_text,y_text-0.8*dy_text,
               '%s mm' % (int(1000*scale_length)),zorder=4)

  def _slider_update(t):
    time_idx = np.argmin(abs(t - times))
    vertical_image[0].remove()
    vert_mask = mask[0][time_idx,:]
    vert_disp = displacement_list[0][time_idx,~vert_mask,2]
    vertical_image[0] = basemap.drawscalar(1000*vert_disp,lonlat[~vert_mask],
                                           cmap=matplotlib.cm.seismic,
                                           zorder=2,vmin=vmin,vmax=vmax,ax=main_ax,alpha=0.5)
    #plt.colorbar(vertical_image[0])
    for idx in range(N):
      if np.any(covariance_list[idx] > 1e-8):
        args = quiver_args(position,
                           displacement_list[idx][time_idx,:,:],
                           covariance_list[idx][time_idx,:,:,:],
                           mask[idx][time_idx,:])
    
        Q_lst[idx].set_UVC(args[2],args[3],sigma=args[4])

      else: 
        args = quiver_args(position,
                           displacement_list[idx][time_idx])
    
        Q_lst[idx].set_UVC(args[2],args[3])

    try:
      # if there is a %s in the title string then try to fill it with a timestamp
      main_ax.set_title(title % modest.decyear_inv(t,format='%Y-%m-%d'))
    except TypeError:
      main_ax.set_title(title)

    main_fig.canvas.draw()
    return

  def _onpick(event):
    idx = np.nonzero(str(event.artist.get_label()) == station_point_label_lst)
    idx = idx[0][0]
    station_lon = lon[idx]
    station_lat = lat[idx]
    station_label = station_names[idx]
    sub_ax[0].cla()
    sub_ax[1].cla()
    sub_ax[2].cla()
    for i in range(N):
      midx = mask[i][:,idx]  
      disp = displacement_list[i][:,idx,:]
      cov = covariance_list[i][:,idx,:,:]
      if np.any(covariance_list[i] > 1e-8):
        sub_ax[0].fill_between(times[~midx],
                           1000*(disp[~midx,0]+np.sqrt(cov[~midx,0,0])),
                           1000*(disp[~midx,0]-np.sqrt(cov[~midx,0,0])),
                               color=colors[i],alpha=0.4,edgecolor='none')
        sub_ax[1].fill_between(times[~midx],
                           1000*(disp[~midx,1]+np.sqrt(cov[~midx,1,1])),
                           1000*(disp[~midx,1]-np.sqrt(cov[~midx,1,1])),
                               color=colors[i],alpha=0.4,edgecolor='none')
        sub_ax[2].fill_between(times[~midx],
                           1000*(disp[~midx,2]+np.sqrt(cov[~midx,2,2])),
                           1000*(disp[~midx,2]-np.sqrt(cov[~midx,2,2])),
                               color=colors[i],alpha=0.4,edgecolor='none')
        sub_ax[0].plot(times[~midx],
                       1000*disp[~midx,0], 
                       colors[i]+'.')
        sub_ax[1].plot(times[~midx],
                       1000*disp[~midx,1], 
                       colors[i]+'.')
        sub_ax[2].plot(times[~midx],
                       1000*disp[~midx,2],
                       colors[i]+'.')
        #sub_ax[0].errorbar(times[~midx],
        #                   1000*disp[~midx,0], 
        #                   1000*np.sqrt(cov[~midx,0,0]),
        #                   color=colors[i],capsize=1,fmt='.')
        #sub_ax[1].errorbar(times[~midx],
        #                   1000*disp[~midx,1], 
        #                   1000*np.sqrt(cov[~midx,1,1]),
        #                   color=colors[i],capsize=1,fmt='.')
        #sub_ax[2].errorbar(times[~midx],
        #                   1000*disp[~midx,2], 
        #                   1000*np.sqrt(cov[~midx,2,2]),
        #                   color=colors[i],capsize=1,fmt='.')
      else:
        sub_ax[0].plot(times[~midx],
                       1000*disp[~midx,0], 
                       colors[i]+'-')
        sub_ax[1].plot(times[~midx],
                       1000*disp[~midx,1], 
                       colors[i]+'-')
        sub_ax[2].plot(times[~midx],
                       1000*disp[~midx,2],
                       colors[i]+'-')


    sub_ax[0].set_title('station %s (%s$^\circ$N, %s$^\circ$E)' % 
                        (station_label,round(station_lat,2),round(station_lon,2)))
    sub_ax[0].legend(disp_type,loc=1,frameon=False,numpoints=4)
    #sub_ax[0].ticklabel_format(useOffset=False, style='plain')
    sub_ax[0].set_ylabel('easting (mm)')
    sub_ax[1].set_ylabel('northing (mm)')
    sub_ax[2].set_ylabel('vertical (mm)')
    sub_ax[2].set_xlabel('year')
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




