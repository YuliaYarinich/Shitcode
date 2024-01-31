# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 08:54:38 2023

@author: kospa
"""

from xrspatial import zonal_stats
from rasterio import features
import geopandas as gpd
import rioxarray
from pyproj import CRS
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

ds = Dataset('F:/new_reanalysis/geo_em.d03.nc')

lons = ds.variables['XLONG_M'][0,:,:]

lats = ds.variables['XLAT_M'][0,:,:]

utci_lcz = ma.array(np.load('E:/ICUC/old/LCZ/utci_2021.npy'), mask = False)

utci_gr = ma.array(np.load('E:/ICUC/old/GR/utci_2021.npy'), mask = False)

utci_wra = ma.array(np.load('E:/ICUC/old/WRa/utci_2021.npy'), mask = False)

utci_ww = ma.array(np.load('E:/ICUC/old/WW/utci_2021.npy'), mask = False)
#%%
import matplotlib.colors as colors

bnds = [-10, -5, -3, -2, -1, -0.1, 0.1, 1, 2, 3, 5, 10]
bnds1 = [-50, -30, -10, -5, -1, 0, 1, 5, 10, 30, 50]
norm = colors.BoundaryNorm(boundaries=bnds1, ncolors=256)
#%%

shapefile = 'C:/Users/kospa/Downloads/atdnew/atd3.shp'
ndvi = rioxarray.open_rasterio('E:/ICUC/time0_T2.tiff').squeeze()
fields = gpd.read_file(shapefile, encoding = 'utf-8')    


fields_utm = fields.to_crs(CRS.from_epsg(4326))
# fields_utm['admin_leve'] = fields_utm['admin_leve'].astype('float32')
fields_utm['inindex'] = range(1, len(fields_utm['name']) + 1)
geom = fields_utm[['geometry', 'inindex']].values.tolist()
fields_rasterized = features.rasterize(geom, out_shape=(201, 201), transform=ndvi.rio.transform())
fields_rasterized_xarr = ndvi.copy()
fields_rasterized_xarr.data = fields_rasterized

k = zonal_stats(fields_rasterized_xarr, np.sum((utci_lcz>=32), axis = 0) - np.sum((utci_ww>=32), axis = 0))
k['inindex'] = k['zone']
k['mean1'] = k['mean']
k = k.drop(labels = ['count', 'zone', 'min', 'mean','max', 'sum', 'std', 'var'], axis = 1)

newdf = fields_utm.merge(k, on = 'inindex', copy = False)
plt.figure(dpi = 256, figsize = (8,6))
ax = plt.axes(projection=ccrs.PlateCarree())
gr=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=False, x_inline=False,
              linewidth=0.5, color='black', alpha=0.7, linestyle="dotted")

gr.top_labels = False
gr.right_labels = False
gr.rotate_labels = False
gr.ylabel_style = {'size': 8, 'color': 'black'}
gr.xlabel_style = {'size': 8, 'color': 'black'}
ax.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)],
              ccrs.PlateCarree(central_longitude=0.0))


newdf.plot(ax = ax, column = 'mean1', 
           transform = ccrs.PlateCarree(), cmap = 'bwr', 
           norm = norm, legend = True)
ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.5, edgecolor='b', alpha=0.7)
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'rivers_europe', '10m'),
                linewidth=0.4, edgecolor='b', facecolor = 'none', alpha = 0.5)
ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'roads', '10m'),
                linewidth=0.2, edgecolor='k',facecolor='none', alpha = 0.3)
shape_feature = ShapelyFeature(Reader('F:/atdrf/admin_level_4.shp').geometries(),
                                ccrs.PlateCarree(), facecolor='none')
ax.add_feature(shape_feature, linewidth=0.7, edgecolor='r', alpha = 0.5)
ax.coastlines('10m', linewidth=0.5, edgecolor = 'k')
plt.title('Разность продолжительности UTCI >= 32℃ \n LCZ - белые крыши и стены 2021')
plt.show()

#%%



