import sys
# Add paths
sys.path.insert(1, './GNC')
sys.path.insert(1, './graphics')
sys.path.insert(1, './postproc')
sys.path.insert(1, './vehicles')
sys.path.insert(1, './algorithm')

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

warnings.filterwarnings('ignore')

# setup simulation
WIDTH = 1200
HEIGHT = 700

CENTERX = WIDTH/2
CENTERY = WIDTH/2

BLACK = (  0,   0,   0)

size = [WIDTH, HEIGHT]
screen = pygame.display.set_mode(size)

# Network
num_of_agents = 4
list_of_agents = []
list_of_edges = [] # For the coordination on the circle

for i in range(num_of_agents-1):
  list_of_edges.append((i,i+1))    # Daisy chain network topology

desired_inter_vehicle_theta = (2*np.pi/num_of_agents)*np.ones((len(list_of_edges),1))
theta_vehicles = np.zeros((num_of_agents,1))
k_coord = 50

# Incidence matrix
B = np.zeros((num_of_agents, len(list_of_edges)))
for idx,edge in enumerate(list_of_edges):
    B[edge[0],idx] =  1
    B[edge[1],idx] = -1

B_dir = np.zeros((num_of_agents, len(list_of_edges)))
for idx,edge in enumerate(list_of_edges):
    B_dir[edge[0],idx] =  1
    B_dir[edge[1],idx] =  0

# Directed vs Undirected graph. They have different convergence properties
# If you want to control with an undirected graph, then uncomment the following.
B_dir = B

for i in range(num_of_agents):
    theta_o = np.pi - (2*np.pi)*np.random.rand(1);
    list_of_agents.append(ag.AgentUnicycle(list(pygame.colordict.THECOLORS.items())[np.random.randint(0,657)][1], i, 1000*np.random.rand(2,1), 60*np.array([[np.cos(theta_o[0])],[np.sin(theta_o[0])]])))

for agent in list_of_agents:
    agent.traj_draw = True

# GVF
ke_circle = 5e-5
kd_circle = 60

xo = 0 # Circle's center
yo = 0
ro = 30
 # radius
stop = 100;
epsilon = 80;
fun = 0;
ck = np.array([xo,yo])
error = np.zeros((2,1),dtype='d')
center = np.array([CENTERX,CENTERY])
counter = 0
counter2 = 0
desv = np.array([1000/np.sqrt(2),1000/np.sqrt(2)])
k = 1000        # Usar si las graficas quedan muy pequeñas el eje y (amplia la escala)

#
direction = 1 # Clock or counter-clock wise. This defines what angular velocity is positive

# run simulation
pygame.init()
clock = pygame.time.Clock()
fps = 50
dt = 1.0/fps
time = 0

runsim = True

pl.figure(1)
x = np.arange(-1000.,2200.,10)
y = np.arange(-700.,1800.,10)
[X,Y] = np.meshgrid(x,y)
ctr_gaussian=center
# Gaussiana de una dimension y sin rotar.
Z = grad.gausianillas(X,Y,ctr_gaussian,[1000/np.sqrt(2),1000/np.sqrt(2)],0,1)
# Tres dimensiones y rotada.
# Z1 = grad.gausianillas(X,Y,ctr_gaussian,[1000/np.sqrt(2),500/np.sqrt(2)],np.pi/6,1)
# ctr_gaussian_2 = np.array([0,1200])
# Z2 = grad.gausianillas(X,Y,ctr_gaussian_2,[300/np.sqrt(2),300/np.sqrt(2)],0,0.9)
# ctr_gaussian_3 = np.array([1200,0])
# Z3 = grad.gausianillas(X,Y,ctr_gaussian_3,[300/np.sqrt(2),300/np.sqrt(2)],0,0.9)
# Z = Z1 + Z2 + Z3
pl.contour(X,Y,Z,np.arange(0,1,0.05))
pl.axis('equal')

while(runsim):
    screen.fill(BLACK)

    us = 0 # We keep constant velocity

    # Coordination algorithm on the circle (calculated in a compact way)
    for idx,agent in enumerate(list_of_agents):
        theta_vehicles[idx] = np.arctan2(agent.pos[1]-yo, agent.pos[0]-xo)

    inter_theta = B.transpose().dot(theta_vehicles)
    error_theta = inter_theta - desired_inter_vehicle_theta

    if np.size(error_theta) > 1:
      for i in range(0, np.size(error_theta)):
        if error_theta[i] > np.pi:
          error_theta[i] = error_theta[i] - 2*np.pi
        elif error_theta[i] <= -np.pi:
          error_theta[i] = error_theta[i] + 2*np.pi
    else:
        if error_theta > np.pi:
          error_theta = error_theta - 2*np.pi
        elif error_theta <= -np.pi:
          error_theta = error_theta + 2*np.pi

    dr = -k_coord*B_dir.dot(np.sin(error_theta))
    dr = direction*dr
    # print("Soy error: ",la.norm(error_theta))
    
    # Algorithm gradient estimation. 
    positions_agents = np.zeros((num_of_agents,2))
    for idx,agent in enumerate(list_of_agents):
        positions_agents[idx,0] =  np.array([agent.pos[0]])     # Probe si era como pasaba las posiciones y lo cambie.
        positions_agents[idx,1] =  np.array([agent.pos[1]])  
    
    # cir = np.zeros((num_of_agents,1),dtype='d')
    # # for k in range(num_of_agents):
    # #     x = positions_agents[k,0]
    # #     y = positions_agents[k,1]
    # #     cir[k] = (x-CENTERX)**2+(y-CENTERY)**2-ro**2
    # for idx,agent in enumerate(list_of_agents):
    #     # positions_agents[idx,0] =  np.array([agent.pos[0]])     # Probe si era como pasaba las posiciones y lo cambie.
    #     # positions_agents[idx,1] =  np.array([agent.pos[1]])  
    #     cir[idx] = (agent.pos[0]-xo)**2+(agent.pos[1]-yo)**2-(ro)**2
        
    # fig = pl.figure(2)
    # ax = fig.add_subplot(111)
    # # pl.plot(counter2,ro**2,'.k',markersize=1) 
    # pl.plot(counter2,cir[0],'.r',markersize=1) 
    # pl.plot(counter2,cir[1],'.g',markersize=1)     
    # pl.plot(counter2,cir[2],'.b',markersize=1)         
    # ax.set_xlabel('N')
    # ax.set_ylabel('f(r)')
    # counter2 = counter2 + 1
        
    # print(positions_agents)
    # # Seria conveniente hacer que avance aca, porque creo que en caso contrario no me hace la animacion
    # # Voy a tener que separar los codigos, uno para calcular el gradiente 
    if (la.norm(error_theta) < 0.2 and fun < 0.999):
        # print("soy xo: ",xo)
        # print("soy yo: ",yo)
        # print("soy fun: ",fun)
        # print("gradestfin: ",gradestfin)
        # print("soy stop: ",stop)
        # print("soy ck: ",ck)
        # print("positions_agents: ",positions_agents)
        gradestfin=grad.computegradient(xo,yo,positions_agents,ro,center,num_of_agents,desv,0,1) # Gradiente (paso el centro para que dentro calcule el valor de la función con el máximo en otro lado, en lugar de en (0,0))
        fun = grad.function(xo-CENTERX,yo-CENTERY,0,desv,1,0) # Gradiente.
        grd = grad.function(xo-CENTERX,yo-CENTERY,0,desv,1,1)
        ck = ck + epsilon*gradestfin # Avance.
        xo = ck[0]
        yo = ck[1]
        if counter%100==0:     
          # print("soy ck: ",ck)
          # print(grd)
          # print("Final: ", gradestfin)
          print("soy fun: ",fun)
          # print("gradestfin: ",gradestfin)
          # print('grd real',grd)
          # fig = pl.figure(1)
          # ax = fig.add_subplot(111)
          # pl.arrow(ck[0],ck[1],k*gradestfin[0],1000*gradestfin[1],color='r')
          #   # pl.arrow(ck[0],ck[1],k*grd[0,0],1000*grd[1,0],color='k')
          # ax.set_xlabel('X')
          # ax.set_ylabel('Y')
         
          # fig = pl.figure(2)
          # ax = fig.add_subplot(111)
          # pl.plot(counter,fun,'.r',markersize=1)         
          # ax.set_xlabel('N')
          # ax.set_ylabel('f(x)')
         
          fig = pl.figure(3)
          ax = fig.add_subplot(111)
          pl.plot(counter,k*grd[0,0],'.r',markersize=1)
          pl.plot(counter,k*grd[1,0],'.b',markersize=1)
          pl.plot(counter,k*gradestfin[0],'.g',markersize=1)
          pl.plot(counter,k*gradestfin[1],'.k',markersize=1)
          ax.legend(['Comp_x_real','Comp_y_real','Comp_x_est','Comp_y_est'])
          ax.set_xlabel('N (iteraciones)')
          ax.set_ylabel(r'$Comparativa:\nabla{f\left(c\right)--}\widehat{\nabla}{f\left(c\right)}$')
    
          # fig = pl.figure(4)
          # ax = fig.add_subplot(111)
          # pl.plot(counter,k*(gradestfin[0]-grd[0,0]),'.r',markersize=1)
          # pl.plot(counter,k*(gradestfin[1]-grd[1,0]),'.b',markersize=1)
          # ax.legend(['Diff_Comp_x','Diff_Comp_y'])
          # ax.set_ylabel('Diferencia componentes del gradiente')
          # ax.set_xlabel('N (iteraciones)')
         
          fig = pl.figure(5)
          ax = fig.add_subplot(111)
          error[0] = gradestfin[0]-grd[0,0]
          error[1] = gradestfin[1]-grd[1,0]
          pl.plot(counter,k*la.norm(error),'.m',markersize=1)
          ax.set_ylabel(r'${||}\widehat{\nabla}{f\left(c\right)-}\nabla{f\left(c\right)}{||}$')
          ax.set_xlabel('N (iteraciones)')
        elif fun >= 0.999:
            break
        counter = counter + 1    
        
    # Guiding vector field
    for idx,agent in enumerate(list_of_agents):
        agent.draw(screen)
        circle_path = gvf.Path_gvf_circle(xo, yo, ro+dr[idx])
        ut = gvf.gvf_control_2D_unicycle(agent.pos, agent.vel, ke_circle, kd_circle, circle_path, direction)
        agent.step_dt(us, ut, dt)

    # clock.tick(fps)
    # pygame.display.flip()

    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         endtime = pygame.time.get_ticks()
    #         pygame.quit()
    #         runsim = False



# # # Postprocessing
# fig = pl.figure(1)
# ax = fig.add_subplot(111)
# for agent in list_of_agents:
#     lines=lp.plot_position(ax, agent)
#     ax.axis("equal")
#     ax.grid
# # ax.legend(lines[:2], ['first', 'second'])
# circle = pl.Circle((xo, yo), ro, color='r')
# ax.add_patch(circle)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# pl.show()