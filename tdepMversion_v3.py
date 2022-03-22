from dolfin import *
from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
 
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
m = 50 # Temporal resolution
dt = T / m
mesh = create_mesh(n)
#plot (mesh)
#plt.show()

#2. Set function space V and trial and test functions u, v
V = FunctionSpace(mesh, 'CG', 2)
v = TestFunction(V)
u = Function(V)
u_n = Function(V)


#3. Define parameters of the problem
cB = Constant(1.0)
mu = Constant(0.01)
f = Constant(0.0)
u_n.interpolate(Constant(0.5))



#4. Define bilinear form
F = ( u.dx(0)*v - mu*u.dx(0)*v.dx(0) - ((u - u_n)/dt)*v - f*v )*dx

#5. Define boundary
def boundary ( x ):
  #Boundaries are x=0 and x=1.0
  value = x[0] < DOLFIN_EPS or 1.0 - DOLFIN_EPS < x[0]
  return value



#6. Solve
t = 0
flux_e = [None]*m
flux_n = [None]*m

 #7. Define exact solution and BC

u_expr = Expression ( "%e*(1-exp(-x[0]/%e)) - (%e/2)*(erf((x[0] + %e)/sqrt(4*%e*%e)) - exp(-x[0]/%e)*(2-erf((x[0] - %e)/sqrt(4*%e*%e))))" % (cB, mu, cB, T, mu, T, mu, T, mu, T), degree = 10 )
  
u_exact = interpolate(u_expr, V)

bc = DirichletBC (V, u_expr, boundary)

for i in range(m):
    
  solve( F == 0 , u, bc)
  u_n.assign(u)
  t += dt

  #8. Calculate the flux of particles which is even harder than the concentration 
#total flux in dimensionless units
  flux = project(mu*u.dx(0)+u,FunctionSpace(mesh, 'CG', 2))
  flux_expr = Expression ( "%e - (%e/2)*(sqrt(%e/(pi*%e))*exp(-(x[0] + %e)*(x[0] + %e)/(4*%e*%e)) + sqrt(%e/(pi*%e))*exp(-x[0]/%e)*exp(-(x[0] - %e)*(x[0] - %e)/(4*%e*%e)) + erf((x[0] + %e)/sqrt(4*%e*%e)))" % (cB, cB, mu, t, t, t, mu, t, mu, t, mu, t, t, mu, t, t, mu, t), degree = 10 )

  flux_exact = interpolate(flux_expr, V)
  flux_vector_e = flux_exact.vector()
  flux_vector_n = flux.vector()
  flux_e[i] = flux_vector_e[0]
  flux_n[i] = flux_vector_n[0]
  
  
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

t = np.linspace(0, T, num = m)
fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot (t, flux_n, label = 'Computed' )
plt.plot (t, flux_e, label = 'Exact' )
ax.legend ( )
ax.grid ( True )
plt.title ( 'Flux in surface, grid %d' % ( n ) )
filename = ( 'Flux_surf_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )
