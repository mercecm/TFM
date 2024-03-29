#! /usr/bin/env python3
#
from dolfin import *
import csv

def mesh(n):
  m = UnitIntervalMesh(n)
  x = m.coordinates()
  x[:] = x[:]**2
  return m

def convection_diffusion ( my_mu, my_grid ):

#*****************************************************************************80
#
## convection_diffusion simulates a 1D convection diffusion problem.
#
#  Discussion:
#
#    - ux - mu uxx = f in Omega = the unit interval.
#      u(0) = 0
#      u(1) = 1
#
#    where 
#
#      mu > 0 is a dimensionless quantity related to physical parameters by 
#             mu = D L/ v (D= diffusion coef, L=size cuvette, v advection velocity)
#     THE VALUE OF MU IS FIXED AT THE BEGINING OF THE MAIN PROGRAM (SEE BELOW)
#
#      u_exact = ( e^(-x/mu) - 1 ) / ( e^(-1/mu) - 1 )
#
#      f(x) = 0
#
#    We are interested in problems where mu is small, dominated by external forcing
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    03 March 2022
#
#  Author:
#
#    John Burkardt
#    Modified by Mercè Clua
#
#  Reference:
#
#    Anders Logg, Kent-Andre Mardal,
#    Lectures on the Finite Element Method.
#
#  Parameters:
#
#    Input, real my_mu, the viscosity.
#    0 < my_mu.
#
#    Input, integer my_grid, the resolution on the unit interval.
#
  import matplotlib.pyplot as plt
#
#  Set the mesh.
#
  print ( '' )
  print ( '  Unit interval mesh n = %d' % ( my_grid ) )
  my_mesh = mesh( my_grid )
#
#  Set the function space.
#
  degree = 2
  V = FunctionSpace ( my_mesh, 'CG', degree )
#
#  Set the trial and test functions.
#
  u = TrialFunction ( V )
  v = TestFunction ( V )
#
#  Make a copy of my_mu.
#
  mu = Constant ( my_mu )
#
#  Set the right hand side.
#
  f = Constant ( 0.0 )
#
#  Define the bilinear form and right hand side
#
  a = ( - u.dx(0) * v + mu * u.dx(0) * v.dx(0) ) * dx
  L = f * v * dx
#
#  Define the exact solution.
#
  u_expr = Expression ( "(exp(-x[0]/%e)-1)/(exp(-1/%e)-1)" % ( my_mu, my_mu ), degree = 10 )
#
#  Define the boundary condition (comment Jordi)
# Take care not to use == as explained in Fenics manual but to allow for tolerance
#
  def boundary ( x ):
#   Boundaries are x=0 and x=1.0
    value = x[0] < DOLFIN_EPS or 1.0 - DOLFIN_EPS < x[0]
    return value

# The boundary condition uses the exact expression that provides automatically u(0)=0 and u(1)=1
# but any other function that has this property will work (comment Jordi)
  bc = DirichletBC ( V, u_expr, boundary )
#
#  Solve.
#
  uh = Function ( V )
#
#  Solve the system.
#
  solve ( a == L, uh, bc )
#
#  Project the exact solution.
#
  u_exact = interpolate ( u_expr, V )

# Calculate the flux of particles which is even harder than the concentration
# by Jordi Faraudo 
  #diffusive flux
  #flux = project(my_mu*uh.dx(0),FunctionSpace(my_mesh, 'CG', 1)) 
  #total flux in dimensionless units
  flux = project(-my_mu*uh.dx(0)-uh, V)
  flux_expr = Constant ( -1.0 )
  flux_exact = interpolate (flux_expr,V)
#
#  Plot the solution (concentration).
#
  fig = plt.figure ( )
  ax = plt.subplot ( 111 )
  plot (uh, label = 'Computed' )
  plot (u_exact, label = 'Exact' )
  ax.legend ( )
  ax.grid ( True )
  plt.title ( 'convection_diffusion solutions, grid %d' % ( my_grid ) )
  filename = ( 'convection_diffusion_solutions_1_grid%d.png' % ( my_grid ) )
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )

#  Plot also the flux of particles 

  fig = plt.figure ( )
  ax = plt.subplot ( 111 )
  plot (flux, label = 'Computed' )
  plot (flux_exact, label = 'Exact (1.0)' )
  ax.legend ( )
  ax.grid ( True )
  plt.title ( 'Flux, grid %d' % ( my_grid ) )
  filename = ( 'Flux_1_grid%d.png' % ( my_grid ) )
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )

# Save also to csv
# by Mercè Clua
  titles = zip('x', 'u', 'ue', 'f', 'fe')
  x_coord = V.tabulate_dof_coordinates()
  everything = zip(x_coord,uh.vector(),u_exact.vector(),flux.vector(),flux_exact.vector())
  f = open('Computed_%d_grid%d.csv' % (degree, my_grid), 'w')
  writer = csv.writer(f, delimiter = '\t')
  writer.writerows(titles)
  writer.writerows(everything)


#  Terminate.
#
  return

def convection_diffusion_test ( ):

#*****************************************************************************80
#
## convection_diffusion_test tests convection_diffusion.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    02 November 2018
#
#  Author:
#
#    John Burkardt
#    Modified Jordi Faraudo 
#
  import dolfin
  import platform
  import time

  print ( time.ctime ( time.time() ) )
# TEST VALUE FOR MU
  my_mu = 0.1
#
#  Report level = only warnings or higher.
#
  level = 30
  set_log_level ( level )

  print ( '' )
  print ( 'Stationary SOlution of convection diffusion Equation:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  FENICS version %s'% ( dolfin.__version__ ) )
  print ( '  Convection/diffusion problem on the unit interval.' )
  print ( '  - ux - mu uxx = f in Omega = the unit interval.' )
  print ( '  u(0) = 0, u(1) = 1' )
  print ( '  mu =',my_mu )

  for my_grid in ( 10, 100):
    my_mu = 0.1
  #my_grid = 10
    convection_diffusion ( my_mu, my_grid )
#
#  Terminate.
#
  print ( '' )
  print ( 'convection_diffusion_test:' )
  print ( '  Normal end of execution.' )
  print ( '' )
  print ( time.ctime ( time.time() ) )
  return

if ( __name__ == '__main__' ):

  convection_diffusion_test ( )

