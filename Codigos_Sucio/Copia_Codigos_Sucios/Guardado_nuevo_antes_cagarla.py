# -*- coding: utf-8 -*-
"""
Created on Tue May 25 23:10:42 2021

@author: tutir
"""

    # x0 = x-c[0]
    # y0 = y-c[1]
    # cth = np.cos(th)
    # sth = np.sin(th)
    # R =np.array([x0,y0])
    # Rot = np.array([[cth,sth],[-sth,cth]]) #matriz de rotación
    # Q = np.array([[1/(2*s[0]**2),0],[0,1/(2*s[1]**2)]])
    # if x0.ndim > 1:       
    #     Rrt = np.dot(R,Rot.transpose(1,0,2))        
    # else:
    #     Rrt = np.dot(R,Rot)
   
    # #V = p*np.exp(-(Rrt[0]**2/(2*s[0]**2)+Rrt[1]**2/(2*s[1]**2)))
    # H = np.sum(Rrt*np.dot(Q,Rrt.transpose(1,0,2)),axis=0)
    # V =p*np.exp(-H)
    # return V

th = np.pi/2
D = 2
angulo = np.linspace(0, 2*np.pi, 1000) 
epsilon = 10
Dimension = 2
robots = 5;
counter = 0;
desv = np.array([1000/np.sqrt(2),1000/np.sqrt(2)])
c = np.array([600,600])
s=np.zeros((2,1),dtype='d')
Hnew=np.zeros((2,2),dtype='d')
s[0]= desv[0]
s[1]= desv[1]
x0 = 100
y0 = 100
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
f = np.exp(-np.sum(H))
    # Dimension = 2;
    # x1 = x0
    # x2 = y0
    # G = np.zeros((Dimension, 1),dtype='d')
    # comp_x = np.zeros((2,2),dtype='d')
    # comp_x[0,0] = x1
    # comp_x[0,1] = x1
    # comp_x[1,0] = x1
    # comp_x[1,1] = x1
    # comp_y = np.zeros((2,2),dtype='d')
    # comp_y[0,0] = x2
    # comp_y[0,1] = x2
    # comp_y[1,0] = x2
    # comp_y[1,1] = x2
    # Res_x = np.divide(Hnew, comp_x,out=np.zeros_like(Hnew), where=comp_x!=0)
    # Res_y = np.divide(Hnew, comp_y,out=np.zeros_like(Hnew), where=comp_y!=0)
    # G[1,0]= (-2*Res_y[1,1]-Res_x[0,1]-Res_x[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) #Primera derivada.
    # G[0,0]= (-2*Res_x[0,0]-Res_y[0,1]-Res_y[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) 