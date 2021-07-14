# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:27:49 2021

@author: tutir
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:00:15 2021

@author: tutir
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
import random

#plt.style.use()
#Este me define el rango y cuantos pasos quiero.
limite = 8
precision = 10
espaciado = limite*2*precision + 1
X = np.linspace(-limite, limite, espaciado)
Y = np.linspace(-limite, limite, espaciado)

# #Prueba para ver las indexaciones en python.
# test = np.array([[1,2],[3,4]])

# #Los defino segun x[i,j] donde i define a la columna mientras que j la fila.
# miro1 = test[0,:]
# miro2 = test[:,0]

Dimension = 2
H = np.zeros((Dimension, Dimension))
H[0,0]=1/25
H[1,1]=1/25

# #Implement]acion de la gausiana.
# rt = np.array([X,Y])
# r = np.transpose(rt)

# #Necesito primero el argumento (esto no es matlab multiplicar matrices requiere bucles medio enfermos).
# parte1 = np.dot(r,H)
# argumento = np.dot(parte1,rt)

#Graficar.
Xgrid, Ygrid = np.meshgrid(X, Y)
#Implement]acion de la gausiana.

# Z = math.e**-(H[0,0]*(Xgrid * Xgrid) + H[1,1]*(Ygrid * Ygrid))
# populate x,y,z arrays

zz = np.zeros((espaciado,espaciado),dtype='d')

for i in range(espaciado):
  for j in range(espaciado):
    zz[i,j] = np.exp(-(H[0,0]*Xgrid[i,j]**2 + H[1,1]*Ygrid[i,j]**2))   
    
       
#Parte del gradiente con la flechita no tienen que ser igual a la x y de antes pero si relacionarse con ella.
#Genero un punto al azar, tal como se hacia en optimziación.

x1 = random.uniform(-limite,limite)
x2 = random.uniform(-limite,limite)
# x1 = 2 
# x2 = 2
xo = np.array([x1,x2])             #Simplemente para tenerlo aca.
f = np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2))      
G = np.zeros((Dimension, 1))
G[1,0]= -2*H[1,1]*x2*np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2))
G[0,0]= -2*H[0,0]*x1*np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2))


# cp = plt.contour(Xgrid, Ygrid, Z,10,'showtext')
# plt.colorbar(cp)
# fig,ax = plt.figure(figsize=(10, 9))
fig, ax = plt.subplots(figsize=(10,9))


# left, bottom, width, height = 1, 1, 1, 1
# ax = fig.add_axes([left, bottom, width, height])


# contours = plt.contour(Xgrid, Ygrid, zz, 10, cmap = 'RdGy')
# plt.clabel(contours, inline=True, fontsize=12)
cp = ax.contour(Xgrid, Ygrid, zz, 10, cmap = 'RdGy')
ax.clabel(cp, inline=True, fontsize=12)
ax.set_title('Contour Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')


#La flechita
plt.quiver(x1,x2,G[0,0],G[1,0],color='b')

 
# Ahora necesito el circulo con los agentes para tener el gradiente aproximado.


# Genero un circulo, aca puedo modificar su radio "D" y saber donde acaban los agentes.


D = 6
h = x1
k = x2
# cir = (D**2 - (X - h)**2)**(1/2) + k

angulo = np.linspace(0, 2*np.pi, 1000)
Xcir = D * np.cos(angulo) + h
Ycir = D * np.sin(angulo) + k

ax.plot(Xcir,Ycir, color='r')


# Ahora los agentes alrededor del circulo, genero uno y el resto mediante la matriz de rotacion.


N = 4
a = 2*np.pi/N
ro = np.array([x1 + D,x2])
cphi = round(np.cos(a),8)
sphi = round(np.sin(a),8)
Rotacion = np.array([[cphi, -sphi] , [sphi, cphi]])
constante = 2/(N*D**2)
rtotal = np.zeros((N,2),dtype='d')

gradest = np.zeros((N,2),dtype='d')
k = 1
ro = ro - xo
for k in range(N):
  a = 2*np.pi*(k)/N
  cphi = round(np.cos(a),8)
  sphi = round(np.sin(a),8)
  Rotacion = np.array([[cphi, -sphi] , [sphi, cphi]])
  ri = np.dot(Rotacion,np.transpose(ro)) 
  # ri = np.dot((ro,Rotacion))
  fradios = np.exp(-(H[0,0]*(ri[0]+xo[0])**2 + H[1,1]*(ri[1]+xo[1])**2))  
  gradest[k-1,0] = fradios*ri[0]
  gradest[k-1,1] = fradios*ri[1]
  rtotal[k-1,0] = ri[0] + xo[0]
  rtotal[k-1,1] = ri[1] + xo[1]
  

# De momento tengo los vectores rotados con respecto al centro de coordenadas
# me falta sumarle justo lo que le reste para poder rotarlos bien.

# rtotal = rtotal + xo

# Con esto ya tengo los N agentes (sean N el caso de prueba son 4 al ser el mas facil de ver a ojo)
# Ahora simplemente pongo la constante y el sumatorio aca para la estimación del gradiente.
  
gradestfin = constante*sum(gradest)
plt.quiver(x1,x2,gradestfin[0],gradestfin[1],color='r')

#  Esta parte comentarla despues, serían los agentes dibujados a ver si estan bien calculados.

ax.plot(rtotal[:,0],rtotal[:,1],'o')

# Todo esto de abajo son pruebas.

#Prueba de que esta bien la funcion y sus flechitas
# A = np.linspace(-limite, limite, round(espaciado/(precision*2)))
# B = np.linspace(-limite, limite, round(espaciado/(precision*2)))

# #Graficar.
# Xg, Yg = np.meshgrid(A, B)
# dx =- 2*H[0,0]*Xg*np.exp(-(H[0,0]*Xg**2 + H[1,1]*Yg**2))
# dy = -2*H[1,1]*Yg*np.exp(-(H[0,0]*Xg**2 + H[1,1]*Yg**2))
# n = 10
# color2 = np.exp(-n*((H[0,0]*dx**2 + H[1,1]*dy**2)))
# ax.quiver(Xg,Yg,dx,dy,color2)

#Muestra todo funciona como un hold on de matlab
plt.show()


#Miro el error (primero meto en un mismo array el gradiente verdadero)
truegradient = np.array([G[0,0],G[1,0]])

error = abs(gradestfin-truegradient)
error

#Hacer despues estudio de esto en una tabla.


#Prueba para ver que me sale la misma grafica que en matlab.

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)



 