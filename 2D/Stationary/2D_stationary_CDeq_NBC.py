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
  x[:,1] = (-2/(beta**2))*(x[:,1]**3)+(3/beta)*(x[:,1]**2)
  return m

#***************************************************************************************************************************#
#SIMULATES A 2D,STATIONARY SOLUTION CONVECTION DIFFUSION PROBLEM with drift only in x and Periodic Boundary Conditions in y #
#***************************************************************************************************************************#


Lx = 0.05
Ly = 0.025
wx = 0.16E-6
wy = 0.0
D = 4.3E-11
mu = D/(Lx*wx)
beta = Ly/Lx
f = 0.0

nx = 50
ny = 50
mesh = create_mesh(nx,ny,beta)

V = FunctionSpace(mesh, 'CG', 2)
v = TestFunction(V)
u = Function(V)

# Mark boundaries

class BoundaryX0(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and x[0] < DOLFIN_EPS

class BoundaryX1(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and 1.0 - DOLFIN_EPS < x[0]

class BoundaryY0(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and x[1] < DOLFIN_EPS

class BoundaryY1(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and beta - DOLFIN_EPS < x[1]

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(9999)
bx0 = BoundaryX0()
bx1 = BoundaryX1()
by0 = BoundaryY0()
by1 = BoundaryY1()
bx0.mark(boundary_markers, 0)
bx1.mark(boundary_markers, 1)
by0.mark(boundary_markers, 2)
by1.mark(boundary_markers, 3)

# Redefine boundary integration measure
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

#Define Dirichlet BC
g = Expression('x[0]', degree = 10)
bc_x0 = DirichletBC (V, g, boundary_markers, 0)
bc_x1 = DirichletBC (V, g, boundary_markers, 1)
bc = [bc_x0, bc_x1]

#Define No-Flux BC
h = Constant(0.0)
integrals_N = [h*v*ds(2), h*v*ds(3)]

F = ( u.dx(0)*v - mu*dot(grad(u), grad(v)) - f*v)*dx + sum(integrals_N)

solve( F == 0 , u, bc)
flux_x = project(-mu*u.dx(0)-u, V)
flux_y = project(-mu*u.dx(1), V)

x_coord = V.tabulate_dof_coordinates()
everything = zip(x_coord[:,0], x_coord[:,1], u.vector(), flux_x.vector(), flux_y.vector())
f = open('CDeqNBC_%d_x_%d.csv' % ( nx, ny ), 'w')
writer = csv.writer(f, delimiter = '\t')
writer.writerows(everything)

 
plt.figure ( )
#ax = plt.subplot ( 111 )
c = plot (u, mode='color', vmin=0, vmax=1)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('Position x\'', fontsize = 12)
plt.ylabel('Position y\'', fontsize = 12)
plt.grid ( True )
plt.title ( 'Concentration, grid %d x %d' % ( nx, ny ) , fontsize = 13)
plt.colorbar(c, orientation = 'horizontal', pad = 0.2)
filename = ( 'CDeqNBC_beta%.1f.png' % ( beta ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.show( )
