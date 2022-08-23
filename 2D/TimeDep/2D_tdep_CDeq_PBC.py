from dolfin import *
from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import csv

def create_mesh(nx,ny,beta):
  m = RectangleMesh(Point(0.0,0.0), Point(1.0,beta), nx, ny)
  x = m.coordinates()
  x[:,0] = x[:,0]**2
  return m

#***************************************************************************************************************************#
#SIMULATES A 2D, TIME DEPENDANT SOLUTION CONVECTION DIFFUSION PROBLEM with drift only in x and Periodic Boundary Conditions in y #
#***************************************************************************************************************************#


class BoundaryX(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[0] < DOLFIN_EPS or x[0] > (1.0 - DOLFIN_EPS)) and on_boundary)
  
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

T = 0.03 # final time
dt = 0.0001
m = int(T/dt) #integer division
Lx = 0.05
Ly = 0.025
wx = 0.16E-6
wy = 0.0
D = 4.3E-11
mu = D/(Lx*wx)
#mu = 0.05
beta = Ly/Lx
f = 0.0

nx = 50
ny = 50
mesh = create_mesh(nx,ny,beta)


V = FunctionSpace(mesh, 'CG', 2, constrained_domain=PeriodicBoundary())
v = TestFunction(V)
u = Function(V)
u_n = Function(V)
u_n.interpolate(Constant(1.0))

boundary1 = BoundaryX()
g = Expression('x[0]', degree = 10)
bc = DirichletBC (V, g, boundary1)

t = 0
x0 = 0.0
x1 = 0.005
x2 = 0.03
i = 0
flux_0 = [[None,None]]*(m+1)
flux_1 = [[None,None]]*(m+1)
flux_2 = [[None,None]]*(m+1)

F = ( u.dx(0)*v - mu*dot(grad(u), grad(v)) - ((u - u_n)/dt)*v - f*v )*dx

f = open('CDeqPBC_%d_x_%d.csv' % ( nx, ny ), 'w')
titles = ['t, x, y, u, fx, fy']
writer = csv.writer(f, delimiter = '\t')
writer.writerows(titles)


while t<T:
  solve( F == 0 , u, bc)
  flux_x = project(-mu*u.dx(0)-u, V)
  flux_y = project(-mu*u.dx(1), V)
  
  x_coord = V.tabulate_dof_coordinates()
  time = [t]*len(x_coord)
  everything = zip(time, x_coord[:,0], x_coord[:,1], u.vector(), flux_x.vector(), flux_y.vector()) 
  writer.writerows(everything)
  
  if i%10 == 0:
    fig = plt.figure ( )
    ax = plt.subplot ( 111 )
    c = plot (u_n, mode='color', vmin=0, vmax=1)
    #ax.legend ( )
    ax.grid ( True )
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.xlabel('Position x', fontsize = 12)
    plt.ylabel('Position y', fontsize = 12)
    plt.title ( 'Concentration, time %.3f' % ( t ), fontsize = 13 )
    plt.colorbar(c, orientation = 'horizontal', pad = 0.2)
    filename = ( 'CDeqPBC_%.3f.png' % ( t ) )
    plt.savefig ( filename )
    #print ( '  Graphics saved as "%s"' % ( filename ) )
    plt.close( )
  
  u_n.assign(u)
  i += 1
  t += dt

 
fig = plt.figure ( )
ax = plt.subplot ( 111 )
c = plot (u, mode='color', vmin=0, vmax=1)
ax.grid ( True )
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('Position x', fontsize = 12)
plt.ylabel('Position y', fontsize = 12)
plt.title ( 'Concentration, grid %d x %d' % ( nx, ny ), fontsize = 13 )
plt.colorbar(c, orientation = 'horizontal', pad = 0.2)
filename = ( 'CDeqPBC_%d_x_%d.png' % ( nx, ny ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close( )

fig = plt.figure ( )
ax = plt.subplot ( 111 )
d = plot (flux_x)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('Position x', fontsize = 12)
plt.ylabel('Position y', fontsize = 12)
plt.title ( 'Computed flux at t = %.2f, grid %d x %d' % ( T, nx, ny ), fontsize = 13 )
plt.colorbar(d)
filename = ( 'Flux_grid_%d_x_%d.png' % ( nx, ny ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

#t = np.linspace(0, T, num = m+1)
#fig = plt.figure ( )
#ax = plt.subplot ( 111 )
#plt.plot (t, flux_0[:,0], label = 'x = %.1f' %x0)
#plt.plot (t, flux_1[:,0], label = 'x = %.3f' %x1)
#plt.plot (t, flux_2[:,0], label = 'x = %.2f' %x2)
#ax.legend ( )
#plt.xlabel('Time')
#plt.ylabel('Flux')
#ax.grid ( True )
#plt.title ( 'Flux in surface, x direction' )
#filename = ( 'Flux_surf_grid_%d_x_%d.png' % ( nx, ny ) )
#plt.savefig ( filename )
#print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.close ( )
