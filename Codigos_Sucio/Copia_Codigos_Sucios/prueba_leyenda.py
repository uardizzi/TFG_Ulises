# -*- coding: utf-8 -*-
"""
Created on Mon May 31 19:51:05 2021

@author: abierto
"""

import numpy as np
import matplotlib.pyplot as pl

x = np.arange(0,10,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x**2)
y4 = np.exp(-x)

pl.plot(x,y1)
pl.plot(x,y2)
pl.plot(x,y3)
pl.plot(x,y4)

pl.legend(['seno','coseno', 'gaussiana','expo'])