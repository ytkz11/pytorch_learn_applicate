#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 5/19/2022 3:33 PM 
# @Author : DKY
# @File : gee_test.py
import ee
import geemap
geemap.set_proxy(port = 56940)
ee.Initialize()

geemap.set_proxy(port = 56940)
ee.Initialize()
# Create a map centered at (lat, lon).
Map = geemap.Map(center=[40, -100], zoom=4)

collection = ee.ImageCollection('LANDSAT/LC08/C01/T1')

point = ee.Geometry.Point(-122.262, 37.8719)
start = ee.Date('2014-06-01')
finish = ee.Date('2019-10-01')

filteredCollection = ee.ImageCollection('LANDSAT/LC08/C01/T1') \
    .filterBounds(point) \
    .filterDate(start, finish) \
    .sort('CLOUD_COVER', True)
a = filteredCollection.getInfo()
a = filteredCollection.toList()
a = 0