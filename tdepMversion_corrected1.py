from dolfin import *
from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
 
def create_mesh(N):
  m = UnitIntervalMesh(N)
  x = m.coordinates()
  x[:] = x[:]**2
  return m
  
  
#****************************************************************
#SIMULATES A 1D,TIME DEPENDANT CONVECTION DIFFUSION PROBLEM
#****************************************************************


#1. Set mesh
n = 50 #Spatial resolution
T = 0.05 # final time
m = 50 # Temporal resolution
dt = T / m
mesh = create_mesh(n)
#plot (mesh)
#plt.show()

#2. Set function space V and trial and test functions u, v
V = FunctionSpace(mesh, 'CG', 2)
u = Function(V)
v = TestFunction(V)
u_n = Function(V)


#3. Define parameters of the problem
cB = Constant(1.0)
mu = Constant(0.01)
f = Constant(0.0)

#4. Define exact solution
#u_est = Expression ( "(exp(-x[0]/%e)-1)/(exp(-1/%e)-1)" % (mu, mu ), degree = 10 )
#u_trans = Expression ("-(1/2)*(erf((x[0] + %e)/sqrt(4*%e*%e)) - exp(-x[0]/%e)*(2-erf((x[0] - %e)/sqrt(4*%e*%e))))" % (T, mu, T, mu, T, mu, T), degree = 10)
#u_expr = u_est + u_trans
u_expr = Expression ( "(exp(-x[0]/%e)-1)/(exp(-1/%e)-1) - (1/2)*(erf((x[0] + %e)/sqrt(4*%e*%e)) - exp(-x[0]/%e)*(2-erf((x[0] - %e)/sqrt(4*%e*%e))))" % (mu, mu, T, mu, T, mu, T, mu, T), degree = 10 )


#5. Define bilinear form
F = ( ((u - u_n)/dt) * v  - u.dx(0) * v + mu * u.dx(0) * v.dx(0) ) * dx
#L = f * v * dx

#6. Define boundary conditions
def boundary ( x ):
  #Boundaries are x=0 and x=1.0
  value = x[0] < DOLFIN_EPS or 1.0 - DOLFIN_EPS < x[0]
  return value

bc = DirichletBC (V, u_expr, boundary)

#7. Solve
t = 0
u_exact = interpolate(u_expr, V)

for i in range(m):
  print(i)
  solve( F == 0, u, bc)
  u_n.assign(u)
  t += dt

#8. Calculate the flux of particles which is even harder than the concentration 
#total flux in dimensionless units
flux = project(mu*u.dx(0)+u,FunctionSpace(mesh, 'CG', 2))
flux_expr = project(mu*u_exact.dx(0)+u_exact,FunctionSpace(mesh, 'CG', 2))

#9. Plot and save output
fig = plt.figure ( )
ax = plt.subplot ( 111 )
plot (u, label = 'Computed' )
plot (u_exact, label = 'Exact' )
ax.legend ( )
ax.grid ( True )
plt.title ( 'convection_diffusion solutions, grid %d' % ( n ) )
filename = ( 'convection_diffusion_solutions_tdep_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
plt.close ( )

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plot (flux, label = 'Computed' )
plot (flux_expr, label = 'Exact (1.0)' )
ax.legend ( )
ax.grid ( True )
plt.title ( 'Flux, grid %d' % ( n ) )
filename = ( 'Flux_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

