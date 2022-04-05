from dolfin import *
from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

 
def create_mesh(N):
  mesh = UnitIntervalMesh(N)
  x = mesh.coordinates()
  x[:] = x[:]**2
  return mesh


#****************************************************************
#SIMULATES A 1D,TIME DEPENDANT CONVECTION DIFFUSION PROBLEM
#****************************************************************


#1. Set mesh
n = 50 #Spatial resolution
T = 0.05 # final time
dt = 0.0001
m = int(T/dt) #integer division
mesh = create_mesh(n)

#2. Set function space V and trial and test functions u, v
V = FunctionSpace(mesh, 'CG', 2)
v = TestFunction(V)
u = Function(V)
u_n = Function(V)

#3. Define parameters of the problem
f = Constant(0.0)
u_n.interpolate(Constant(1.0))

L = Constant(0.05)
w = Constant(0.16E-6)
D = Constant(4.3E-11)
mu = Expression("%e/(%e*%e)" %(D,L,w), degree = 10) 

#4. Define bilinear form
F = ( u.dx(0)*v - mu*u.dx(0)*v.dx(0) - ((u - u_n)/dt)*v - f*v )*dx

#5. Define boundary and BC
def boundary ( x ):
  #Boundaries are x=0 and x=1.0
  value = x[0] < DOLFIN_EPS or 1.0 - DOLFIN_EPS < x[0]
  return value
  
f = Expression("x[0]*x[0]", degree = 10)
bc = DirichletBC (V, f, boundary)

#6. Solve
t = 0
x0 = 0.0
x1 = 0.005
x2 = 0.03
i = 0
flux_0 = [None]*m
flux_1 = [None]*m
flux_2 = [None]*m

while t<T:
  solve( F == 0 , u, bc)
  flux = project(mu*u.dx(0)+u,FunctionSpace(mesh, 'CG', 2))

  u_n.assign(u)
  flux_0[i] = flux(x0)
  flux_1[i] = flux(x1)
  flux_2[i] = flux(x2)
  i += 1
  t += dt
  
  
#7. Plot and save output

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plot (u, label = 'Computed' )
#ax.legend ( )
ax.grid ( True )
plt.xlabel('Position')
plt.ylabel('Concentration')
plt.title ( 'Computed concentration, grid %d' % ( n ) )
filename = ( 'convection_diffusion_solutions_tdep_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plot (flux, label = 'Computed' )
#ax.legend ( )
ax.grid ( True )
plt.xlabel('Position')
plt.ylabel('Flux')
plt.title ( 'Computed flux at t = %.2f, grid %d' % ( T, n ) )
filename = ( 'Flux_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

t = np.linspace(0, T, num = m)
fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot (t, flux_0, label = 'x = %.1f' %x0)
plt.plot (t, flux_1, label = 'x = %.3f' %x1)
plt.plot (t, flux_2, label = 'x = %.2f' %x2)
ax.legend ( )
plt.xlabel('Time')
plt.ylabel('Flux')
ax.grid ( True )
plt.title ( 'Flux in surface' )
filename = ( 'Flux_surf_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )
