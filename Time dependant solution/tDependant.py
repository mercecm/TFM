from dolfin import *
from mshr import *
from fenics import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
from scipy.special import erfc
import csv

 
def create_mesh(N):
  mesh = UnitIntervalMesh(N)
  x = mesh.coordinates()
  x[:] = x[:]**4
  return mesh


#****************************************************************
#SIMULATES A 1D,TIME DEPENDANT CONVECTION DIFFUSION PROBLEM
#****************************************************************


#1. Set mesh
n = 50 #Spatial resolution
T = 0.75 # final time
dt = 0.0001#/50
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

L = 0.05#/50
w = 0.16E-6
D = 4.3E-11
mu = D/(L*w)

#4. Define bilinear form
F = ( u.dx(0)*v - mu*u.dx(0)*v.dx(0) - ((u - u_n)/dt)*v - f*v )*dx

#5. Define boundary and BC
def boundary ( x ):
  #Boundaries are x=0 and x=1.0
  value = x[0] < DOLFIN_EPS or 1.0 - DOLFIN_EPS < x[0]
  return value
  
g = Expression('x[0]', degree = 10)
bc = DirichletBC (V, g, boundary)

#6. Solve
t = 0

x0 = 0.0
x1 = 0.05
x2 = 0.3

t0 = 0.0001
t1 = 0.0025
t2 = 0.0050

u_0 = Function(V)
u_1 = Function(V)
u_2 = Function(V)

i = 0
flux_0 = [None]*(m+1)
flux_1 = [None]*(m+1)
flux_2 = [None]*(m+1)

titles = zip('t', 'x', 'u', 'f')
f = open('Computed_mu%.3f.csv' % (mu), 'w')
writer = csv.writer(f, delimiter = '\t')
writer.writerows(titles)


while t<T:
  solve( F == 0 , u, bc)
  flux = project(-mu*u.dx(0)-u,V)
  
  if t <= t0:
    u_0.assign(u)
    f_0 = flux
#  elif t <= t1:
#    u_1.assign(u)
#    f_1 = flux
  elif t <= t2:
    u_2.assign(u)
    f_2 = flux

  u_n.assign(u)
  flux_0[i] = -flux(x0)
  flux_1[i] = -flux(x1)
  flux_2[i] = -flux(x2)
  i += 1
  t += dt
  
  
#7. Plot and save output
print(mu)

x_coord = V.tabulate_dof_coordinates()
everything = zip(x_coord[:,0], u_0.vector(), f_0.vector(), u_2.vector(), f_2.vector(), u.vector(), flux.vector()) 
writer.writerows(everything)

#plot concentration
fig = plt.figure ( )
ax = plt.subplot ( 111 )
plot (u_0, label = 't = %.4f' %t0 )
#plot (u_1, label = 't = %.4f' %t1 )
plot (u_2, label = 't = %.3f' %t2 )
plot (u, label = 't = %.2f' %t )
ax.legend ( )
plt.xlabel('Position')
plt.ylabel('Concentration')
plt.title ( 'Computed concentration, grid %d' % ( n ) )
plt.ylim(-0.25,1.25)
filename = ( 'convection_diffusion_solutions_tdep_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )
'''
fig = plt.figure ( )
ax = plt.subplot ( 111 )
plot (f_0, label = 't = %.4f' %t0 )
#plot (f_1, label = 't = %.4f' %t1 )
plot (f_2, label = 't = %.3f' %t2 )
plot (flux, label = 't = %.2f' %t )
ax.legend ( )
plt.xlabel('Position')
plt.ylabel('Flux')
plt.title ( 'Computed flux at t = %.2f, grid %d' % ( T, n ) )
filename = ( 'Flux_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )
'''
t = np.linspace(0, T, num = m+1)
fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot (t, flux_0, label = 'x = %.1f' %x0)
plt.plot (t, flux_1, label = 'x = %.2f' %x1)
plt.plot (t, flux_2, label = 'x = %.1f' %x2)
ax.legend (fontsize = 12 )
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('Time, t\'', fontsize = 12)
plt.ylabel('Flux, J', fontsize = 12)
plt.title ( 'Flux in surface' )
plt.ylim(0,14)
filename = ( 'Flux_surf_grid%d.png' % ( n ) )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )
