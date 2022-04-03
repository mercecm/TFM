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
  
def c_exact(x,t,mu):
  c = (1-np.exp(-x/mu)) - (1/2)*(erfc((x+t)/(np.sqrt(4*mu*t))) - np.exp(-x/mu)*(2-erfc((x-t)/(np.sqrt(4*mu*t)))))
  return c

def c_stationary(x,mu):
  c = (1-np.exp(-x/mu))
  return c
  
def flux_exact(x,t,mu):
  f_e = 1
  f_t = (-1/2)*(np.sqrt(mu/(np.pi*t))*np.exp(-(x+t)*(x+t)/(4*mu*t)) - np.sqrt(mu/(np.pi*t))*np.exp(-x/mu)*np.exp(-(x-t)*(x-t)/(4*mu*t)) + erfc((x+t)/np.sqrt(4*mu*t)))
  f = f_e + f_t
  return f

def flux_stationary(x,mu):
  f = np.exp(x-x)
  return f

def flux_surf(t,mu):
  f = 1-(1/2)*erfc(np.sqrt(t/(4*mu)))
  return f

#****************************************************************
#SIMULATES A 1D,TIME DEPENDANT CONVECTION DIFFUSION PROBLEM
#****************************************************************


#1. Set mesh
n = 100 #Spatial resolution
T = 0.05 # final time
dt = 0.001
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
flux_n = [None]*m
i = 0

while t<T:
  solve( F == 0 , u, bc)
  u_n.assign(u)

  flux = project(mu*u.dx(0)+u,FunctionSpace(mesh, 'CG', 2))

  flux_vector_n = flux.vector()
  flux_n[i] = flux_vector_n[0]
  i += 1
  t += dt
  
  
#7. Plot and save output
xe = np.linspace(0,1,1000)

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plot (u, label = 'Computed' )
#plt.plot (xe, c_exact(xe,T,mu), label = 'Exact' )
ax.legend ( )
ax.grid ( True )
plt.title ( 'convection_diffusion solutions, grid %d' % ( n ) )
filename = ( 'convection_diffusion_solutions_tdep_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plot (flux, label = 'Computed' )
#plt.plot (xe, flux_exact(xe,T,mu), label = 'Exact' )
ax.legend ( )
ax.grid ( True )
plt.title ( 'Flux at t = %e, grid %d' % ( T, n ) )
filename = ( 'Flux_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

t = np.linspace(0, T, num = m)
fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot (t, flux_n, label = 'Computed' )
#plt.plot (t, flux_surf(t,mu), label = 'Exact' )
ax.legend ( )
ax.grid ( True )
plt.title ( 'Flux in surface, grid %d' % ( n ) )
filename = ( 'Flux_surf_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )
