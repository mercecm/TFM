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
#SIMULATES A 2D,STATIONARY SOLUTION CONVECTION DIFFUSION PROBLEM with drift only in x and Periodic Boundary Conditions in y #
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

boundary1 = BoundaryX()
g = Expression('x[0]', degree = 10)
bc = DirichletBC (V, g, boundary1)

F = ( u.dx(0)*v - mu*dot(grad(u), grad(v)) - f*v )*dx

solve( F == 0 , u, bc)
flux_x = project(-mu*u.dx(0)-u, V)
flux_y = project(-mu*u.dx(1), V)

x_coord = V.tabulate_dof_coordinates()
everything = zip(x_coord[:,0], x_coord[:,1], u.vector(), flux_x.vector(), flux_y.vector())
f = open('CDeqPBC_%d_x_%d.csv' % ( nx, ny ), 'w')
writer = csv.writer(f, delimiter = '\t')
writer.writerows(everything)

 
plt.figure ( )
#ax = plt.subplot ( 111 )
c = plot (u, mode='color', vmin=0, vmax=1)
#ax.legend( )
plt.grid ( True )
plt.title ( 'Concentration, grid %d x %d' % ( nx, ny ) )
plt.colorbar(c, orientation = 'horizontal', pad = 0.2)
filename = ( 'CDeqPBC_beta%.1f.png' % ( beta ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.show( )
