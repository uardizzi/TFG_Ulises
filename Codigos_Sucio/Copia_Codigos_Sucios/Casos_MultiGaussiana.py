# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 01:38:19 2021

@author: tutir
"""

# Recordar modificar en funcion de lo que llame en el pdf.
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
import random
import time
from numpy import linalg as la
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

# Cosas para graficar.
# limite = 
# precision = 10
# espaciado = limite*2*precision + 1
# X = np.linspace(-100., 1300., 10)
# Y = np.linspace(-100., 1300., 10)
X = np.arange(-1200.,2400.,10)
Y = np.arange(-1200.,2400.,10)
Xgrid, Ygrid = np.meshgrid(X, Y)

# Parametros
p = 1
rota = 0
D = 80
angulo = np.linspace(0, 2*np.pi, 1000) 
epsilon = 2000
Dimension = 2
robots = 5;
counter = 0;
# desv = np.array([1000/np.sqrt(2),800/np.sqrt(2)])
desv = np.array([1000/np.sqrt(2),500/np.sqrt(2)])
c = np.array([600,600])

# Puntos de donde parte mi centro (esto posteriormente lo paso desde su codigo)
x1 = 0
x2 = 800
ck = np.array([x1,x2])             #Ck en cada momento sera el centro.

def gausianillas(x,y,c,s,th,p):
    """
    Devuelve el valor de una gausiana definida en dos dimesiones x e y.
    Cada dimension tiene su propia desviación. La gausisana puede tener su
    centro en cualquier valor arbitrario y estar orientada 
    también de modo arbitrario
    x e y pueden ser:
    un par de puntos en los que calcular el valor V que toma la gausiana
    Un par de matrices generadas por meshgrid para crear una malla de puntos
    En este caso V es una matriz de la misma dimension que x e y con los
    valores que toma v en los  nodos de la malla
    c =[cx,cy]. representa el centro de la gausiana debe ser un array de 
    dos elementos
    s = [sx,sy] desviación en la dirección xy, en lo que podríamos llamar
    'ejes gaussiana'
    th angulo de los 'ejes gaussiana' respecto a los ejes tierra.
    p modulo de la gausiana (valor que toma en el centro)
    
    Return -> V (mirar definición en el código de la función línea 41)
    """

    x0 = x-c[0]
    y0 = y-c[1]
    cth = np.cos(th)
    sth = np.sin(th)
    R =np.array([x0,y0])
    Rot = np.array([[cth,sth],[-sth,cth]]) #matriz de rotación
    Q = np.array([[1/(2*s[0]**2),0],[0,1/(2*s[1]**2)]])
    if x0.ndim > 1:       
        Rrt = np.dot(R,Rot.transpose(1,0,2))        
    else:
        Rrt = np.dot(R,Rot)
   
    #V = p*np.exp(-(Rrt[0]**2/(2*s[0]**2)+Rrt[1]**2/(2*s[1]**2)))
    H = np.sum(Rrt*np.dot(Q,Rrt.transpose(1,0,2)),axis=0)
    V =p*np.exp(-H)
    return V



def function(x1,x2,th,desv,p,flag):
    s=np.zeros((2,1),dtype='d')
    Hnew=np.zeros((2,2),dtype='d')
    s[0]= desv[0]
    s[1]= desv[1]
    x0 = x1
    y0 = x2
    cth = np.cos(th)
    sth = np.sin(th)
    R =np.array([x0,y0])
    Rot = np.array([[cth,sth],[-sth,cth]]) #matriz de rotación
    Q = np.array([[1/(2*s[0]**2),0],[0,1/(2*s[1]**2)]])
    Rrt = np.dot(R,Rot)
    H = Rrt*Q*Rrt
    Hnew[0,0] = H[0,0]
    Hnew[0,1] = H[0,1]
    Hnew[1,0] = H[1,0]
    Hnew[1,1] = H[1,1]
    Hfin = np.sum(H)
    f = p*np.exp(-np.sum(H))
    Dimension = 2;
    x1 = x0
    x2 = y0
    G = np.zeros((Dimension, 1),dtype='d')
    comp_x = np.zeros((2,2),dtype='d')
    comp_x[0,0] = x1
    comp_x[0,1] = x1
    comp_x[1,0] = x1
    comp_x[1,1] = x1
    comp_y = np.zeros((2,2),dtype='d')
    comp_y[0,0] = x2
    comp_y[0,1] = x2
    comp_y[1,0] = x2
    comp_y[1,1] = x2
    Res_x = np.divide(Hnew, comp_x,out=np.zeros_like(Hnew), where=comp_x!=0)
    Res_y = np.divide(Hnew, comp_y,out=np.zeros_like(Hnew), where=comp_y!=0)
    G[1,0]= (-2*Res_y[1,1]-Res_x[0,1]-Res_x[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) #Primera derivada.
    G[0,0]= (-2*Res_x[0,0]-Res_y[0,1]-Res_y[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) 
    # print("meh: ",G[1,0])
    # print("H: ",Hnew)
    # print("H: ",Res_y)
    if flag==0:
        return f
    else:
        return G  

zz = np.zeros((len(X),len(Y)),dtype='d')
zz1 = np.zeros((len(X),len(Y)),dtype='d')
zz2 = np.zeros((len(X),len(Y)),dtype='d')
zz3 = np.zeros((len(X),len(Y)),dtype='d')
for i in range(len(X)):
  for j in range(len(Y)):
    zz1[i,j] = function(Xgrid[i,j]-c[0],Ygrid[i,j]-c[1],rota-np.pi/6,desv,p,0) 
    zz2[i,j] = function(Xgrid[i,j],Ygrid[i,j]-1200,rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0) 
    zz3[i,j] = function(Xgrid[i,j]-1200,Ygrid[i,j],rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0) 
    
zz = zz1+zz3+zz2
#Prueba para ver que me sale la misma grafica que en matlab, simplemente es la grafica en tres dimensiones.
# Z = gausianillas(Xgrid, Ygrid,c,desv,0,1)

def placerobots(x1,x2,robots):
    xo = np.array([x1,x2]) 
    N = robots
    a = 2*np.pi/N
    ro = np.array([x1 + D,x2])
    rtotal = np.zeros((N,2),dtype='d')
    ro = ro - xo
    for k in range(N):
        if k == N-1:
            ri = np.array([0,0])
        else:
            a = 2*np.pi*(k+1)/(N-1)
            cphi = round(np.cos(a),8)
            sphi = round(np.sin(a),8)
            Rotacion = np.array([[cphi, -sphi] , [sphi, cphi]])
            ri = np.dot(Rotacion,np.transpose(ro)) 
        rtotal[k,0] = ri[0] + xo[0]
        rtotal[k,1] = ri[1] + xo[1]
    return rtotal
    
  
def computegradient(x1,x2,robots,positions,desv,c,rota,p):
    xo = np.array([x1,x2]) 
    N = robots-1
    constante = 2/(N*D**2)
    gradest = np.zeros((N,2),dtype='d')
    for k in range(N):
        f1 = function(positions[k,0]-c[0],positions[k,1]-c[1],rota-np.pi/6,desv,p,0) 
        f2 = function(positions[k,0],positions[k,1]-1200,rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0) 
        f3 = function(positions[k,0]-1200,positions[k,1],rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0)
        fradios = f1 + f2 + f3
        gradest[k,0] = fradios*(positions[k,0]-x1)
        gradest[k,1] = fradios*(positions[k,1]-x2)
    gradestfin = constante*sum(gradest)
    return gradestfin

counter_t = np.zeros((200,1),dtype='d')
error = np.zeros((200,1),dtype='d')
positions = np.zeros((robots,2),dtype='d')
positions = placerobots(x1,x2,robots)
fun = 0
fig, ax = plt.subplots(figsize=(7,5))
while(fun < 0.99999):
    plt.figure(1)
    plt.pause(0.1) # Espera.
    plt.cla()
    cp = ax.contour(Xgrid, Ygrid, zz, 30, cmap = 'RdGy') # Contorno.
    ax.clabel(cp, inline=True, fontsize=12)
    # ax.set_title('Contour Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    h = ck[0]   # Circulito
    k = ck[1]
    Xcir = D * np.cos(angulo) + h
    Ycir = D * np.sin(angulo) + k
    ax.plot(Xcir,Ycir, color='r')
    gradestfin=computegradient(ck[0],ck[1],robots,positions,desv,c,rota,p) # Gradiente.
    ax.plot(positions[:robots-1,0],positions[:robots-1,1],'o')
    plt.quiver(ck[0],ck[1],gradestfin[0],gradestfin[1],color='r')
    positions = positions + epsilon*gradestfin
    ck = ck + epsilon*gradestfin # Avance.
    fun1 = function(ck[0]-c[0],ck[1]-c[1],rota-np.pi/6,desv,p,0)
    fun2 = function(ck[0],ck[1]-1200,rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0)
    fun3 = function(ck[0]-1200,ck[1],rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0)
    fun = fun1 + fun2 + fun3
    print("soy fun: ",fun)
    # print("gradestfin: ",gradestfin)
    # print("soy stop: ",stop)
    # print("soy ck: ",ck)
    # gr_real = function(ck[0]-c[0],ck[1]-c[1],rota,desv,p,1)
    if sum(abs(gradestfin)) <= 10**-4 : # Condicion de parada.
        break
    
    
# fig = plt.figure(2)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Plot the surface.
# surf = ax.plot_surface(Xgrid, Ygrid, zz, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x,y)')



 