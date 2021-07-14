# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 01:34:20 2021

@author: tutir
"""

from time import sleep
import pygame
import matplotlib.pyplot as pl
import matplotlib.colors as mcolors
import drawmisc
import agents as ag
import numpy as np
from numpy import linalg as la
import gradiente as grad
import logpostpro as lp
import gvf
import warnings

N = 600
fig = pl.figure(1)
ax = fig.add_subplot(111)
for k in range(N):
    pl.plot(k,k,'.b',markersize=2)