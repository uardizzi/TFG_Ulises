# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:48:27 2021

@author: tutir
"""
s=np.zeros((2,1),dtype='d')
Hnew=np.zeros((2,2),dtype='d')
s[0]= 707
s[1]= 707
x0 = 600
y0 = 0
th = 0
cth = np.cos(th)
sth = np.sin(th)
R =np.array([x0,y0])
Rot = np.array([[cth,sth],[-sth,cth]]) #matriz de rotación
Q = np.array([[1/(2*s[0]**2),0],[0,1/(2*s[1]**2)]])
Rrt = np.dot(Rot,R)
H = Rrt*(Q*Rrt)
Hnew[0,0] = H[0,0]
Hnew[0,1] = H[0,1]
Hnew[1,0] = H[1,0]
Hnew[1,1] = H[1,1]
Hfin = np.sum(H)
f = np.exp(-np.sum(H))
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
G[1,0]= (-2*Res_y[0,0]-Res_x[0,1]-Res_x[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) #Primera derivada.
G[0,0]= (-2*Res_x[0,0]-Res_y[0,1]-Res_y[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) 

comparar = (-2*1/(2*s[1]**2)*x1*np.exp(-(1/(2*s[1]**2)*x1**2 + 1/(2*s[1]**2)*x2**2)) )