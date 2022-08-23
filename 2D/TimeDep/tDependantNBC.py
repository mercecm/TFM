from dolfin import *
from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
#from scipy.special import erfc
import csv

 
def create_mesh(nx,ny,beta):
  m = RectangleMesh(Point(0.0,0.0), Point(1.0,beta), nx, ny)
  x = m.coordinates()
  x[:,0] = x[:,0]**2
  x[:,1] = (-2/(beta**2))*(x[:,1]**3)+(3/beta)*(x[:,1]**2)
  return m

#***************************************************************************************************************************#
#SIMULATES A 2D,TIME SOLUTION CONVECTION DIFFUSION PROBLEM with drift only in x and Periodic Boundary Conditions in y #
#***************************************************************************************************************************#


T = 5 # final time
dt = 0.0001
m = int(T/dt) #integer division
Lx = 0.05
Ly = 0.1
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
plot(mesh)

V = FunctionSpace(mesh, 'CG', 2)
v = TestFunction(V)
u = Function(V)
u_n = Function(V)
u_n.interpolate(Constant(1.0))


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

# Mark boundaries
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
g = Expression('x[0]*x[0]', degree = 10)
bc_x0 = DirichletBC (V, g, boundary_markers, 0)
bc_x1 = DirichletBC (V, g, boundary_markers, 1)
bc = [bc_x0, bc_x1]

#Define Neumann BC
integrals_N = [-u*v*ds(2), -u*v*ds(3)]


#Define bilinear form
F = ( u.dx(0)*v - mu*dot(grad(u), grad(v)) - ((u - u_n)/dt)*v - f*v )*dx + sum(integrals_N)


#Solve
t = 0
x0 = 0.0
x1 = 0.005
x2 = 0.03
i = 0
flux_0 = [[None,None]]*(m+1)
flux_1 = [[None,None]]*(m+1)
flux_2 = [[None,None]]*(m+1)

f = open('CDeqNBC_%d_x_%d.csv' % ( nx, ny ), 'w')
titles = ['t, x, y, u, flux']
writer = csv.writer(f, delimiter = '\t')
writer.writerows(titles)

diff = 1

while t<T:
  solve( F == 0 , u, bc)
  flux_x = project(-mu*u.dx(0)-u, V)
  flux_y = project(-mu*u.dx(1), V)
  d = project(u-u_n, V)
  diff = float(d.vector().max())
  u_n.assign(u)
  flux_0[i] = -flux_x((x0,0.0))
  flux_1[i] = -flux_x((x1,0.0))
  flux_2[i] = -flux_x((x2,0.0))
  
  x_coord = V.tabulate_dof_coordinates()
  time = [t]*len(x_coord)
  everything = zip(time, x_coord[:,0], x_coord[:,1], u.vector(), flux_x.vector(), flux_y.vector()) 
  writer.writerows(everything)
  if i%500 == 0:
    fig = plt.figure ( )
    ax = plt.subplot ( 111 )
    c = plot (u, mode='color', vmin=0, vmax=1)
    ax.legend ( )
    #ax.grid ( True )
    plt.title ( 'Concentration, time %.3f' % ( t ) )
    plt.colorbar(c)
    filename = ( 'CDeqNBC_%.3f.png' % ( t ) )
    plt.savefig ( filename )
    #print ( '  Graphics saved as "%s"' % ( filename ) )
    plt.close( )

    fig = plt.figure ( )
    ax = plt.subplot ( 111 )
    d = plot (flux_x)
    ax.legend ( )
    #ax.grid ( True )
    plt.title ( 'Flux, time %.3f' % ( t ) )
    plt.colorbar(d)
    filename = ( 'Flux-CDeqNBC_%.3f.png' % ( t ) )
    plt.savefig ( filename )
    #print ( '  Graphics saved as "%s"' % ( filename ) )
    plt.close( )
  
  i += 1
  t += dt
  
  
#7. Plot and save output

fig = plt.figure ( )
ax = plt.subplot ( 111 )
c = plot (u)
#ax.legend ( )
#ax.grid ( True )
plt.xlabel('Position')
plt.ylabel('Concentration')
plt.title ( 'Computed concentration, grid %d x %d' % ( nx, ny ) )
plt.colorbar(c)
filename = ( 'convection_diffusion_solutions_tdep_grid%d_x_%d.png' % ( nx, ny ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

fig = plt.figure ( )
ax = plt.subplot ( 111 )
d = plot (flux_x)
#ax.legend ( )
#ax.grid ( True )
plt.xlabel('Position')
plt.ylabel('Flux')
plt.title ( 'Computed flux at t = %.2f, grid %d x %d' % ( T, nx, ny ) )
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
