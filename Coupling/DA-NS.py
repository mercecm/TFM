from dolfin import *
from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import csv

 #***************************************************************************************************************************#
#SIMULATES A 2D, CONVECTION DIFFUSION SYSTEM WHOSE VELOCITY FIELD IS OBTAINED THROUGH NAVIER-STOKES EQUATIONS UNDER A MAGNETIC GRADIENT #
#***************************************************************************************************************************#

#Parameters
nx = 50
ny = 50

T = 0.005 # final time
dt = 1E-5
m = int(T/dt) #integer division

Lx = 0.05
Ly = 0.025
beta = Ly/Lx
D = 4.3E-11
rhonp = 4.86E3
V = (4/3)*np.pi*(12E-9)*(12E-9)*(12E-9)/8 #(30.94E-9)*(30.94E-9)*(30.94E-9)/8
R = (12E-9)/2


c0 = 1E-2
M = 68#42.7#*4*np.pi*1E-10
rho = 997
eta = 1E-3
#p0 = 0.00000005*101325/(rho*wx*wx)
p0 = 1

gradB_v = 30E-8 #100
gradB = -30E-8 #-100 #1.45
#gradB = Expression('-30*(1-x[0])', degree = 10)

#wx = 0.16E-6
wx = (V*rhonp*M*gradB_v)/(6*np.pi*eta*R)
ext_f = (Lx*c0*M*gradB)/(wx*wx)

mu = D/(Lx*wx)
Re = (rho*wx*Lx)/eta


#Functions for mesh and Boundaries

def create_mesh(nx,ny,beta):
  m = RectangleMesh(Point(0.0,0.0), Point(1.0,beta), nx, ny)
  x = m.coordinates()
  x[:,0] = x[:,0]**2
  x[:,1] = (-2/(beta**2))*(x[:,1]**3)+(3/beta)*(x[:,1]**2)
  return m

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

#Mesh and function spaces
mesh = create_mesh(nx,ny,beta)
#plot(mesh)
#plt.show()

V = FunctionSpace(mesh, 'P', 2) #function space for concentration 
W = VectorFunctionSpace(mesh, 'P', 2) #function space for velocity
Q = FunctionSpace(mesh, 'P', 1) #function space for pressure

#Trial and test functions
u = Function(V) #concentration
u_n = Function(V) #concentration at previous time step
v = TestFunction(V) #test for concentration

w = Function(W) #velocity field
w_n = Function(W)
z = TestFunction(W)
w2 = w_n + Constant((-1.0,0.0))

p = Function(Q) #pressure
p_n = Function(Q)
q = TestFunction(Q)


#set initial conditions
u_n.interpolate(Constant(1.0))
w_n.interpolate(Constant((0.0,0.0)))

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

#Define Dirichlet BC of DA
g = Expression('x[0]', degree = 10)
bc_x0 = DirichletBC (V, g, boundary_markers, 0)
bc_x1 = DirichletBC (V, g, boundary_markers, 1)
bc_DA = [bc_x0, bc_x1]

#Define Neumann BC of DA
integrals_N = [-w2[1]*u*v*ds(2), -w2[1]*u*v*ds(3)]

#Diffusion-advection equation

F1 = (div(u*w2)*v + mu*dot(grad(u), grad(v)) + ((u - u_n)/dt)*v)*dx + sum(integrals_N)

#Define Dirichlet BC of NS

#bc_inflow = DirichletBC(Q, p0, boundary_markers, 0)
bc_outflow = DirichletBC(Q, p0, boundary_markers, 1)

bc_noslip0 = DirichletBC(W, Constant((0,0)), boundary_markers, 0)
bc_noslip1 = DirichletBC(W, Constant((0,0)), boundary_markers, 2)
bc_noslip2 = DirichletBC(W, Constant((0,0)), boundary_markers, 3)

bc_w = [bc_noslip0, bc_noslip1, bc_noslip2]
bc_p = [bc_outflow]

#Second order splitting algorithm for Navier-Stokes eqs
n = FacetNormal(mesh) #normal vector to mesh
f = as_vector([u*ext_f, 0.0]) #external magnetic field

# Define strain-rate tensor
def epsilon(w):
    return sym(nabla_grad(w))

# Define stress tensor
def sigma(w, p):
    return 2*(1/Re)*epsilon(w) - p*Identity(len(w))

F2 = dot((w - w_n) / dt, z)*dx + dot(dot(w_n, nabla_grad(w_n)), z)*dx + inner(sigma(0.5*(w + w_n), p_n), epsilon(z))*dx + dot(p_n*n, z)*ds - (1/Re)*dot(nabla_grad(0.5*(w + w_n))*n,z)*ds - dot(f, z)*dx #- dot(dot(sigma(0.5*(w + w_n), p_n), n), z)*ds

F3 = dot(grad(p), grad(q))*dx - dot(grad(p_n), grad(q))*dx + (1/dt)*div(w)*q*dx

F4 = dot(w,z)*dx - dot(w_n,z)*dx + dt*dot(grad(p-p_n),z)*dx

#Solve

header = ['t', 'x', 'y', 'u', 'flux x', 'flux y', 'wx', 'wy']

out = open('magnetophoresis.csv', 'w')
writer = csv.writer(out, delimiter = '\t')
writer.writerow(header)

t = 0
i = 0

while t<T:

	w2 = w_n + Constant((-1.0,0.0))
	solve(F1 == 0, u, bc_DA)

	solve(F2 == 0, w, bc_w) #half time step velocity computed , solver_parameters={"newton_solver":{"relative_tolerance":5e-8}} 
	
	solve(F3 == 0, p, bc_p) #
	
	w_n.assign(w) #half time step velocity used as previous time step to compute new time step velocity
	
	solve(F4 == 0, w, bc_w)
	
	if i%10 == 0:
		fig = plt.figure ( )
		ax = plt.subplot ( 111 )
		c = plot (u_n, mode='color', vmin=0, vmax=1)
		ax.grid ( True )
		plt.xticks(fontsize = 11)
		plt.yticks(fontsize = 11)
		plt.xlabel('Position x', fontsize = 12)
		plt.ylabel('Position y', fontsize = 12)
		plt.title ( 'Concentration, time %.1f' % ( i/10 ), fontsize = 13 )
		plt.colorbar(c, orientation = 'horizontal', pad = 0.2)
		filename = ( 'couple_%.1f.png' % ( i/10 ) )
		plt.savefig ( filename )
		plt.close( )
			
		flux = project(-mu*grad(u) + u*w_n, W)
		flux_x,flux_y = flux.split(deepcopy = True)
		
		w_x,w_y = w_n.split(deepcopy = True)

		#plot(flux, mode = 'glyphs')
		#plt.show()
		
		#plot(w, mode = 'glyphs')
		#plt.show()
		
		x_coord = V.tabulate_dof_coordinates()
		time = [t]*len(x_coord)
		everything = zip(time, x_coord[:,0], x_coord[:,1], u.vector(), flux_x.vector(), flux_y.vector(), w_x.vector(), w_y.vector()) 
		writer.writerows(everything)
  
	p_n.assign(p)
	#w_n.assign(w)
	u_n.assign(u)

	i += 1
	t += dt

out.close()
	
flux = project(-mu*grad(u) + u*w, W)
flux_x,flux_y = flux.split(deepcopy = True)

w_x,w_y = w.split(deepcopy = True)

#output

final = open('magnetophoresis_finalT.csv', 'w')
writer = csv.writer(final, delimiter = '\t')
writer.writerow(header)
x_coord = V.tabulate_dof_coordinates()
time = [t]*len(x_coord)
everything = zip(time, x_coord[:,0], x_coord[:,1], u.vector(), flux_x.vector(), flux_y.vector(), w_x.vector(), w_y.vector()) 
writer.writerows(everything)
final.close()

fig = plt.figure ( )
ax = plt.subplot ( 111 )
c = plot (u, mode='color', vmin=0, vmax=1)
ax.grid ( True )
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('Position x', fontsize = 12)
plt.ylabel('Position y', fontsize = 12)
plt.title ( 'Concentration, time %.3f' % ( t ), fontsize = 13 )
plt.colorbar(c, orientation = 'horizontal', pad = 0.2)
filename = ( 'couple_final.png' )
plt.savefig ( filename )
plt.close( )

print(mu)
print(Re)
print(wx)




