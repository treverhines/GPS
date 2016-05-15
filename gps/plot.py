#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import myplot.cm
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon
from myplot.basemap import Basemap
from myplot.quiver import Quiver
from myplot.colorbar import transparent_colorbar
import misc
import matplotlib
import modest
from rbf.geometry import contains

matplotlib.quiver.Quiver = Quiver # for error ellipses
mygreen = (0.0,0.7,0.0)

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

def record_section(data_list,
                   epicenter,
                   radius_range, 
                   angle_range,
                   name_list=None,
                   basemap=None, 
                   colors=None,
                   mapscale_lonlat=None,
                   map_resolution=100,  
                   minimap_pos=None):

  if colors is None:
    colors = ['k','b','g','y','c','b','y']

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
    
  #times = data_list[0]['time'][:]
  lon = data_list[0]['position'][:,0]
  lat = data_list[0]['position'][:,1]
  station_names = data_list[0]['name'][:]  
  print(station_names)

  if basemap is None:
    basemap = create_default_basemap(lat,lon)


  
  # form polygon which encloses stations to include in the record section
  epicenter_xy = basemap(*epicenter)
  theta = np.linspace(angle_range[0],angle_range[1],100)
  x_inner = epicenter_xy[0] + radius_range[0]*np.cos(theta)
  y_inner = epicenter_xy[1] + radius_range[0]*np.sin(theta)
  x_outer = epicenter_xy[0] + radius_range[1]*np.cos(theta[::-1])
  y_outer = epicenter_xy[1] + radius_range[1]*np.sin(theta[::-1])
  x = np.concatenate((x_inner,x_outer))
  y = np.concatenate((y_inner,y_outer))
  xy = np.array([x,y]).T
  smp = np.array([np.arange(xy.shape[0]),np.roll(np.arange(xy.shape[0]),-1)]).T
  stax,stay = basemap(lon,lat)
  staxy = np.array([stax,stay]).T
  include = np.nonzero(contains(staxy,xy,smp))[0]

  basemap = create_default_basemap(basemap(x,y,inverse=True)[1],basemap(x,y,inverse=True)[0])
  epicenter_xy = basemap(*epicenter)
  theta = np.linspace(angle_range[0],angle_range[1],100)
  x_inner = epicenter_xy[0] + radius_range[0]*np.cos(theta)
  y_inner = epicenter_xy[1] + radius_range[0]*np.sin(theta)
  x_outer = epicenter_xy[0] + radius_range[1]*np.cos(theta[::-1])
  y_outer = epicenter_xy[1] + radius_range[1]*np.sin(theta[::-1])
  x = np.concatenate((x_inner,x_outer))
  y = np.concatenate((y_inner,y_outer))
  xy = np.array([x,y]).T
  smp = np.array([np.arange(xy.shape[0]),np.roll(np.arange(xy.shape[0]),-1)]).T
  stax,stay = basemap(lon,lat)
  staxy = np.array([stax,stay]).T
  #include = np.nonzero(contains(staxy,xy,smp))[0]

  rs_fig = plt.figure('Record Section',figsize=(4.0,6.66))
  rs_ax = rs_fig.add_axes([0.2,0.1,0.7,0.8])
  
  fig = plt.figure('Map View',figsize=(4,7))
  ax = fig.add_axes([0.1,0.1,0.8,0.8])

  ax.plot(stax[include],stay[include],'ko',zorder=3)
  staxy = staxy[include]
  station_names = station_names[include]
  for i in range(len(staxy)):
    if station_names[i] in ['P603','P501']:
      ax.text(staxy[i,0]-25000,staxy[i,1]-12000,station_names[i],zorder=3)
    elif station_names[i] in ['I40A']:
      ax.text(staxy[i,0]+2000,staxy[i,1]-12000,station_names[i],zorder=3)
    elif station_names[i] in ['P499','OPBL']:
      ax.text(staxy[i,0]-27000,staxy[i,1]+3000,station_names[i],zorder=3)
    else:
      ax.text(staxy[i,0]+2000,staxy[i,1]+2000,station_names[i],zorder=3)

  # add record section polygon
  poly = Polygon(xy,closed=True,color='blue',alpha=0.4,edgecolor='none',zorder=2)
  ax.add_artist(poly)

  basemap.drawcoastlines(ax=ax,linewidth=2.0,zorder=1,color=(0.3,0.3,0.3,1.0))
  basemap.drawcountries(ax=ax,linewidth=2.0,zorder=1,color=(0.3,0.3,0.3,1.0))
  basemap.drawstates(ax=ax,linewidth=2.0,zorder=1,color=(0.3,0.3,0.3,1.0))
  basemap.drawmeridians(np.arange(np.floor(basemap.llcrnrlon),
                        np.ceil(basemap.urcrnrlon),1.0),
                        labels=[0,0,0,1],dashes=[2,2],
                        ax=ax,zorder=1,color=(0.3,0.3,0.3,1.0))
  basemap.drawparallels(np.arange(np.floor(basemap.llcrnrlat),
                        np.ceil(basemap.urcrnrlat),1.0),
                        labels=[1,0,0,0],dashes=[2,2],
                        ax=ax,zorder=1,color=(0.3,0.3,0.3,1.0))

  #basemap.drawtopography(ax=ax,vmin=-6000,vmax=4000,
  #                       alpha=0.4,resolution=map_resolution,zorder=0)

  basemap.drawmapscale(units='km',
                         lat=basemap.latmin+(basemap.latmax-basemap.latmin)/15.0,
                         lon=basemap.lonmin+(basemap.lonmax-basemap.lonmin)/4.0,
                         fontsize=8,
                         lon0=(basemap.lonmin+basemap.lonmax)/2.0,
                         lat0=(basemap.latmin+basemap.latmax)/2.0,
                         barstyle='fancy',ax=ax,
                         length=50,zorder=4)
  #stax,stay = basemap(lon,lat)
  #staxy = np.array([stax,stay]).T
  #smp = np.array([np.arange(xy.shape[0]),np.roll(np.arange(xy.shape[0]),-1)]).T
  #include = np.nonzero(contains(staxy,xy,smp))[0]

  # angle by which the displacement components need to be rotated to get radial component
  rotation_angle = np.arctan2(staxy[:,1]-epicenter_xy[1],staxy[:,0]-epicenter_xy[0])
  zeros = np.zeros(len(staxy))
  ones =  np.ones(len(staxy))
  rotation_matrices = np.array([[ np.cos(rotation_angle),np.sin(rotation_angle),zeros],
                                [-np.sin(rotation_angle),np.cos(rotation_angle),zeros],
                                [zeros,                  zeros,                 ones ]])
  rotation_matrices = np.einsum('ijk->kij',rotation_matrices)

  def H(t):
    return (t>=0.0).astype(float)

  def logsys(m,t):
    return m[0]*np.log(1 + H(t-2010.257)*(t-2010.257)*m[1])

  def expsys(m,t):
    return m[0]*(1 - np.exp(-H(t-2010.257)*(t-2010.257)*m[1]))

  ts_width = 1.0
  for idx,d in enumerate(data_list):
    # normalize all displacements so that the width is ts_width
    # rotate displacements to get radial component
    times = d['time'][:]
    mask = np.array(d['mask'][:,include],dtype=bool)
    disp = np.einsum('...ij,...j->...i',rotation_matrices,d['mean'][:,include,:])[...,0]
    #disp = disp[...] - disp[0,...]
    var = np.einsum('...ij,...jk,...lk->...il',rotation_matrices,d['covariance'][:,include,:,:],rotation_matrices)[...,0,0]
    #var = d['covariance'][:,include,1,1]
    std = np.sqrt(var)

    print(disp[0,:])
    if idx == 0:
      scale = ts_width/(np.max(disp,axis=0) - np.min(disp,axis=0))
      shift = np.copy(disp[0,:])

    disp -= shift
    disp *= scale
    std *= scale   
    dist = np.sqrt((staxy[:,0]-epicenter_xy[0])**2 + (staxy[:,1]-epicenter_xy[1])**2)/1000
    order = np.argsort(dist)
    #dist = np.argsort(dist)*ts_width
    dy = 0
    ytickpos = []
    yticklabel = []
    for i in range(1):
      ytickpos += [dy] 
      yticklabel += [''] 
      dy += 1.0*ts_width

    for i in order:
      if np.any(d['covariance'][...] > 1e-8):
        rs_ax.fill_between(times[~mask[:,i]],
                           disp[~mask[:,i],i]+std[~mask[:,i],i]+dy,
                           disp[~mask[:,i],i]-std[~mask[:,i],i]+dy,
                           color=colors[idx],alpha=0.4,edgecolor='none')
      #pred1 = modest.nonlin_lstsq(expsys,disp[:,i],2,system_args=(times,),output=['predicted'])
      #pred2 = modest.nonlin_lstsq(logsys,disp[:,i],2,system_args=(times,),output=['predicted'])
      rs_ax.plot(times[~mask[:,i]],disp[~mask[:,i],i]+dy,colors[idx]+'-',lw=1)
      ytickpos += [dy] 
      dy += 1.0*ts_width
      #rs_ax.plot(times,pred1+dist[i],'b-')
      #rs_ax.plot(times,pred2+dist[i],'r-')

  min_time = np.min([np.min(d['time'][:]) for d in data_list])
  max_time = np.max([np.max(d['time'][:]) for d in data_list])
  times = np.linspace(min_time,max_time,100)
  ytickpos = np.array(ytickpos)
  station_names = np.array(['%s\n(%s km)'%(i,int(j)) for (i,j) in zip(station_names,dist)])
  yticklabel = np.concatenate((yticklabel,station_names[order]))
  xtickpos = np.arange(np.floor(np.min(times)),np.ceil(np.max(times)))
  xticklabel = np.array([str(int(i)) for i in xtickpos])
  plt.sca(rs_ax)
  plt.yticks(ytickpos,yticklabel,fontsize=8)
  plt.xticks(xtickpos,xticklabel,fontsize=8)
  rs_ax.grid()
  #rs_ax.ticklabel_format(useOffset=False, style='plain')
  rs_ax.set_xlabel('year',fontsize=8)
  rs_ax.set_ylabel('station (epicentral distance)',fontsize=8)
  rs_ax.set_ylim(np.min(ytickpos)-0.1,np.max(ytickpos)+0.1)
  rs_ax.set_xlim(np.min(times)-0.1,np.max(times)+0.1)

  plt.show()
  return



def view(data_list,
         name_list=None,
         units=None,
         draw_map=True,
         basemap=None, 
         title='displacement at %s',
         init_time=None,
         quiver_scale=0.00001,
         quiverscale_lonlat=None,
         mapscale_lonlat=None,
         mapscale_length=100, 
         minimap_pos=None,  
         vmin=-20,vmax=20,
         scale_length=1.0,
         ax=None,
         artists=None):
  # data list is a list of dictionary-like objects with keys: mean,
  # covariance, mask, position, time
  mask_list = []
  mean_list = []
  cov_list = []
  time_list = []
  if name_list is None:
    name_list = ['displacement %s' % i for i in range(len(data_list))]

  for data in data_list:
    mask_list += [data['mask'][...]]
    mean_list += [data['mean'][...]]
    cov_list += [data['covariance'][...]]
    time_list += [data['time'][...]]
    
  times = data_list[0]['time'][:]
  lon = data_list[0]['position'][:,0]
  lat = data_list[0]['position'][:,1]
  station_names = data_list[0]['name'][:]

  if artists is None:
    artists = []

  _view(mean_list,
        cov_list,
        time_list,
        station_names,
        lon,
        lat,
        mask_list,
        init_time=init_time,
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
        scale_length=scale_length,
        mapscale_length=mapscale_length,
        ax=ax,
        artists=artists)

def _view(displacement_list,
          covariance_list,
          time_list,
          station_names,
          lon,
          lat,
          mask,
          init_time=None,
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
          mapscale_length=100,
          ax=None, 
          artists=None):

  lonlat = np.array([lon,lat]).T
  #times -= 2010.0
  N = len(displacement_list)
  Nx = len(lon)
  if disp_type is None:
    disp_type = ['']*N
  
  disp_type = [i.replace('_',' ') for i in disp_type]
  if colors is None:
    colors = ['k',mygreen,'r','b','m']

  if artists is None:
    artists = []

  sub_fig,sub_ax = plt.subplots(3,1,figsize=(8,6),sharex=True)
  # do not use scientific notation
  sub_ax[0].ticklabel_format(useOffset=False, style='plain')

  slider_fig = plt.figure('time slider',figsize=(8,1))
  slider_ax = slider_fig.add_axes([0.1,0.1,0.8,0.8])
  time_slider = Slider(slider_ax,'time',
                       min(time_list[0]),max(time_list[0]),
                       valinit=min(time_list[0]),
                       color='black')

  if ax is None:
    main_fig = plt.figure('Map View',figsize=(6,3))
    main_ax = main_fig.add_axes([0.1,0.1,0.8,0.8])
  else:
    main_ax = ax
    main_fig = ax.figure
  
  if init_time is not None:
    time = init_time
  else:
    time = min(time_list[0])

  try:
    # if there is a %s in the title string then try to fill it with a timestamp
    main_ax.set_title(title % modest.decyear_inv(time,format='%Y-%m-%d'),fontsize=10)
  except TypeError:
    main_ax.set_title(title,fontsize=10)
  
  #time_idx = np.argmin(abs(time - time))

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
    #basemap.drawtopography(ax=main_ax,vmin=-6000,vmax=4000,
    #                       alpha=0.4,resolution=map_resolution,zorder=0)

    # zorder 1
    basemap.drawcoastlines(ax=main_ax,linewidth=1.0,zorder=1,color=(0.3,0.3,0.3,1.0))
    basemap.drawcountries(ax=main_ax,linewidth=1.0,zorder=1,color=(0.3,0.3,0.3,1.0))
    basemap.drawstates(ax=main_ax,linewidth=1.0,zorder=1,color=(0.3,0.3,0.3,1.0))
    basemap.drawmeridians(np.arange(np.floor(basemap.llcrnrlon),
                          np.ceil(basemap.urcrnrlon),1.0),
                          labels=[0,0,0,1],dashes=[2,2],
                          ax=main_ax,zorder=1,color=(0.3,0.3,0.3,1.0),fontsize=8,linewidth=0.5)
    basemap.drawparallels(np.arange(np.floor(basemap.llcrnrlat),
                          np.ceil(basemap.urcrnrlat),1.0),
                          labels=[1,0,0,0],dashes=[2,2],
                          ax=main_ax,zorder=1,color=(0.3,0.3,0.3,1.0),fontsize=8,linewidth=0.5)

    # zorder 0
    time_idx = np.argmin(abs(time - time_list[0]))
    vert_mask = mask[0][time_idx,:]
    vert_disp = displacement_list[0][time_idx,~vert_mask,2]
    vertical_image = [basemap.drawscalar(1000*vert_disp,lonlat[~vert_mask],
                                         cmap=matplotlib.cm.RdBu_r,
                                         zorder=0,vmin=vmin,vmax=vmax,ax=main_ax)]

    # zorder 4
    if mapscale_lonlat is None:
      basemap.drawmapscale(units='km',
                         lat=basemap.latmin+(basemap.latmax-basemap.latmin)/3.25,
                         lon=basemap.lonmax-(basemap.lonmax-basemap.lonmin)/7.75,
                         fontsize=8,
                         lon0=(basemap.lonmin+basemap.lonmax)/2.0,
                         lat0=(basemap.latmin+basemap.latmax)/2.0,
                         barstyle='fancy',ax=main_ax,
                         length=mapscale_length,zorder=4)
    else:
      basemap.drawmapscale(units='km',
                           lat=mapscale_lonlat[1],
                           lon=mapscale_lonlat[0],
                           fontsize=8,
                           lon0=(basemap.lonmin+basemap.lonmax)/2.0,
                           lat0=(basemap.latmin+basemap.latmax)/2.0,
                           barstyle='fancy',ax=main_ax,
                           length=mapscale_length,zorder=4)

    #cbar = transparent_colorbar(vertical_image[0])
    #cbar.set_alpha(1)
    #cbar.draw_all()
    cbar = plt.colorbar(vertical_image[0])
    cbar.ax.tick_params(labelsize=8)
    cbar.solids.set_rasterized(True)
    cbar.ax.set_ylabel('vertical (mm)',fontsize=8)
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
    station_point = main_ax.plot(x,y,'ko',markersize=0.01,picker=8,zorder=3)
    station_point_label_lst += [station_point[0].get_label()]
    station_point_lst += station_point

  station_point_label_lst = np.array(station_point_label_lst,dtype=str)

  Q_lst = []
  for idx in range(N):
    time_idx = np.argmin(abs(time - time_list[idx]))
    print(np.min(1000*displacement_list[idx][time_idx,:,2])) 
    print(np.max(1000*displacement_list[idx][time_idx,:,2])) 
    if idx >= 1:
      main_ax.scatter(position[:,0],position[:,1],c=1000*displacement_list[idx][time_idx,:,2],s=50,linewidths=1,
                      cmap=matplotlib.cm.RdBu_r,vmin=vmin,vmax=vmax,edgecolor=mygreen,zorder=2)
      
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
                               ellipse_linewidths=1.0,
                               zorder=4+idx)]
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
                               zorder=4+idx)]

  #v_scale = np.array([0.0])
  #z_scale = np.array([0.0])
  xlim = main_ax.get_xlim()
  ylim = main_ax.get_ylim()
  xo = xlim[1]
  yo = ylim[1]
  xwidth = xlim[1] - xlim[0]
  ywidth = ylim[1] - ylim[0]
  if quiverscale_lonlat is None:
    x_scale = xo - 2.25*xwidth/10.0
    y_scale = yo - 4.4*ywidth/10.0
  else:
    x_scale,y_scale = basemap(*quiverscale_lonlat)

  main_ax.quiverkey(Q_lst[0],x_scale,y_scale,scale_length,coordinates='data',labelsep=0.05,
                    fontproperties={'size':8},label='%s mm' % int(1000*scale_length))

  #x_text = x_scale
  #y_text = y_scale - 0.5*ywidth/10.0 
  #dy_text = -ywidth/10.0

  #for i in range(N):
  #  main_ax.text(x_text,y_text+i*dy_text,disp_type[i])
  #  main_ax.quiver(x_scale,y_scale+i*dy_text,u_scale,v_scale,
  #                 scale_units='xy',
  #                 angles='xy',
  #                 width=0.004,
  #                 scale=quiver_scale,
  #                 color=colors[i],zorder=4)

  #main_ax.text(x_text,y_text-0.8*dy_text,
  #             '%s mm' % (int(1000*scale_length)),zorder=4)

  def _slider_update(t):
    time_idx = np.argmin(abs(t - time_list[0]))
    vertical_image[0].remove()
    vert_mask = mask[0][time_idx,:]
    vert_disp = displacement_list[0][time_idx,~vert_mask,2]
    vertical_image[0] = basemap.drawscalar(1000*vert_disp,lonlat[~vert_mask],
                                           cmap=matplotlib.cm.RdBu_r,
                                           zorder=0,vmin=vmin,vmax=vmax,ax=main_ax)
    #plt.colorbar(vertical_image[0])
    for idx in range(N):
      time_idx = np.argmin(abs(t - time_list[idx]))
      if idx >= 1:
        main_ax.scatter(position[:,0],position[:,1],c=1000*displacement_list[idx][time_idx,:,2],s=50,linewidths=1,
                        cmap=matplotlib.cm.RdBu_r,vmin=vmin,vmax=vmax,edgecolor=mygreen,zorder=2)

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
      main_ax.set_title(title % modest.decyear_inv(t,format='%Y-%m-%d'),fontsize=10)
    except TypeError:
      main_ax.set_title(title,fontsize=10)

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
        sub_ax[0].fill_between(time_list[i][~midx],
                           1000*(disp[~midx,0]+np.sqrt(cov[~midx,0,0])),
                           1000*(disp[~midx,0]-np.sqrt(cov[~midx,0,0])),
                               color=colors[i],alpha=0.4,edgecolor='none')
        sub_ax[1].fill_between(time_list[i][~midx],
                           1000*(disp[~midx,1]+np.sqrt(cov[~midx,1,1])),
                           1000*(disp[~midx,1]-np.sqrt(cov[~midx,1,1])),
                               color=colors[i],alpha=0.4,edgecolor='none')
        sub_ax[2].fill_between(time_list[i][~midx],
                           1000*(disp[~midx,2]+np.sqrt(cov[~midx,2,2])),
                           1000*(disp[~midx,2]-np.sqrt(cov[~midx,2,2])),
                               color=colors[i],alpha=0.4,edgecolor='none')
        sub_ax[0].plot(time_list[i][~midx],
                       1000*disp[~midx,0], 
                       color=colors[i],linestyle='-')
        sub_ax[1].plot(time_list[i][~midx],
                       1000*disp[~midx,1], 
                       color=colors[i],linestyle='-')
        sub_ax[2].plot(time_list[i][~midx],
                       1000*disp[~midx,2],
                       color=colors[i],linestyle='-')
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
        sub_ax[0].plot(time_list[i][~midx],
                       1000*disp[~midx,0], 
                       color=colors[i],linestyle='-')
        sub_ax[1].plot(time_list[i][~midx],
                       1000*disp[~midx,1], 
                       color=colors[i],linestyle='-')
        sub_ax[2].plot(time_list[i][~midx],
                       1000*disp[~midx,2],
                       color=colors[i],linestyle='-')


    sub_ax[0].set_title('station %s (%s$^\circ$N, %s$^\circ$E)' % 
                        (station_label,round(station_lat,2),round(station_lon,2)))
    sub_ax[0].legend(disp_type,loc=1,frameon=False,numpoints=4)
    sub_ax[0].ticklabel_format(useOffset=False, style='plain')
    sub_ax[0].set_ylabel('easting (mm)')
    sub_ax[1].set_ylabel('northing (mm)')
    sub_ax[2].set_ylabel('vertical (mm)')
    sub_ax[2].set_xlabel('year')
    sub_fig.canvas.draw()
    event.artist.set_markersize(10)
    main_fig.canvas.draw()
    event.artist.set_markersize(0.01)
    return

  time_slider.on_changed(_slider_update)
  main_fig.canvas.mpl_connect('pick_event',_onpick)

  for a in artists:
    main_ax.add_artist(a)  

  plt.show()




