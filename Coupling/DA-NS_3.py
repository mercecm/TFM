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
#Parameters
nx = 50
ny = 50

T = 0.03 # final time
dt = 1E-4
m = int(T/dt) #integer division

Lx = 0.05
Ly = 0.025
beta = Ly/Lx
D = 4.3E-11
rhonp = 4.86E3
V = (4/3)*np.pi*(12E-9)*(12E-9)*(12E-9)/8
R = (12E-9)/2


c0 = 1E-2
M = 68
rho = 867 #997
eta = 5E-4

gradB_v = 30
gradB = -30
#gradB = Expression('-30*(1-x[0])', degree = 10)

wx = (V*rhonp*M*gradB_v)/(6*np.pi*eta*R)
wf = wx*60
ext_f = (c0*M*gradB)/(rho*wf*wf)

mu = D/(Lx*wx)
Re = (rho*wf*Lx)/eta

print(wx)
print(mu)
print(Re)


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
u = TrialFunction(V) #concentration
u_ = Function(V)
u_n = Function(V) #concentration at previous time step
v = TestFunction(V) #test for concentration

w = TrialFunction(W) #velocity field
w_n = Function(W)
w_ = Function(W)
z = TestFunction(W)
#w2 = Function(W)

p = TrialFunction(Q) #pressure
p_n = Function(Q)
p_ = Function(Q)
q = TestFunction(Q)


#set initial conditions
u_n.interpolate(Constant(1.0))
w_n.interpolate(Constant((0.0,0.0)))
#w2 = w_n + Constant((-1.0,0.0))
w_m = interpolate(Expression(('-(1-x[0])', '0.0'), degree = 10), W)
w2 = w_n + w_m

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
#g = Expression('x[0]', degree = 10)
bc_x0 = DirichletBC(V, 0, boundary_markers, 0)
#bc_x1 = DirichletBC(V, 1, boundary_markers, 1)
bcD = [bc_x0] #, bc_x1

#Define Neumann BC of DA
n = FacetNormal(mesh) #normal vector to mesh

integrals_N = [-w2[1]*u*v*ds(2), -w2[1]*u*v*ds(3), -w2[0]*u*v*ds(1)]

#Diffusion-advection equation

F1 = (div(u*w2)*v + mu*dot(grad(u), grad(v)) + ((u - u_n)/dt)*v)*dx + sum(integrals_N)
a1 = lhs(F1)
L1 = rhs(F1)


#Define Dirichlet BC of NS

#bc_outflow = DirichletBC(Q, p0, boundary_markers, 0)
#bc_inflow = DirichletBC(Q, p0, boundary_markers, 1)

bc_noslip0 = DirichletBC(W, Constant((0,0)), boundary_markers, 0)
bc_noslip1 = DirichletBC(W, Constant((0,0)), boundary_markers, 2)
bc_noslip2 = DirichletBC(W, Constant((0,0)), boundary_markers, 3)
bc_noslip3 = DirichletBC(W, Constant((0,0)), boundary_markers, 1)

bcw = [bc_noslip0, bc_noslip1, bc_noslip2, bc_noslip3]
#bcp = [bc_inflow]

#Second order splitting algorithm for Navier-Stokes eqs

f = as_vector([u_*ext_f, 0.0]) #external magnetic field

# Define strain-rate tensor
def epsilon(w):
    return sym(nabla_grad(w))

# Define stress tensor
def sigma(w, p):
    return 2*(1/Re)*epsilon(w) - p*Identity(len(w))
dtf = dt*60

F2 = dot((w - w_n) / dtf, z)*dx + dot(dot(w_n, nabla_grad(w_n)), z)*dx + inner(sigma(0.5*(w + w_n), p_n), epsilon(z))*dx + dot(p_n*n, z)*ds - (1/Re)*dot(nabla_grad(0.5*(w + w_n))*n,z)*ds - dot(f, z)*dx - dot(dot(sigma(0.5*(w + w_n), p_n), n), z)*ds
a2 = lhs(F2)
L2 = rhs(F2)


a3 = dot(grad(p), grad(q))*dx 
L3 = dot(grad(p_n), grad(q))*dx - (1/dtf)*div(w_)*q*dx

a4 = dot(w,z)*dx 
L4 = dot(w_n,z)*dx - dtf*dot(grad(p_-p_n),z)*dx


#Solve

header = ['t', 'x', 'y', 'u', 'wx', 'wy', 'wx plot', 'wy plot'] #'flux x', 'flux y', 

out = open('magnetophoresis.csv', 'w')
writer = csv.writer(out, delimiter = '\t')
writer.writerow(header)

t = 0
i = 0

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A4 = assemble(a4)


[bc.apply(A1) for bc in bcD]
[bc.apply(A2) for bc in bcw]
#[bc.apply(A3) for bc in bcp]

while t<T:

#	w2 = (60)*w_n + Constant((-1.0,0.0))
	w2 = w_n + w_m
	b1 = assemble(L1)
	[bc.apply(b1) for bc in bcD]
	solve(A1, u_.vector(), b1)
	
	if i%60 == 0:
		b2 = assemble(L2)
		[bc.apply(b2) for bc in bcw]
		solve(A2, w_.vector(), b2)
	
		b3 = assemble(L3)
		solve(A3, p_.vector(), b3)
	
		w_n.assign(w_) #half time step velocity used as previous time step to compute new time step velocity
	
		b4 = assemble(L4)
		solve(A4, w_.vector(), b4)
		p_n.assign(p_)
		w_n.assign(w_)
	
	u_n.assign(u_)

	
	if i%10 == 0:
		fig = plt.figure ( )
		ax = plt.subplot ( 111 )
		c = plot (u_n, mode='color', vmin=0, vmax=1)
		ax.grid ( True )
		plt.xticks(fontsize = 11)
		plt.yticks(fontsize = 11)
		plt.xlabel('Position x', fontsize = 12)
		plt.ylabel('Position y', fontsize = 12)
		plt.title ( 'Concentration, time step %.4f' % ( t ), fontsize = 13 )
		plt.colorbar(c, orientation = 'horizontal', pad = 0.2)
		filename = ( 'couple_%.4f.png' % ( t ) )
		plt.savefig ( filename )
		plt.close( )
			
		#flux = project(-mu*grad(u) + u*w_n, W)
		#flux_x,flux_y = flux.split(deepcopy = True)
		
		w_x,w_y = w_n.split(deepcopy = True)

		#plot(flux, mode = 'glyphs')
		#plt.show()
		
		#plot(w, mode = 'glyphs')
		#plt.show()
		
		x_coord = V.tabulate_dof_coordinates()
		time = [t]*len(x_coord)
		everything = zip(time, x_coord[:,0], x_coord[:,1], u_.vector(),  w_x.vector(), w_y.vector(), w_x.vector()*1E-9, w_y.vector()*1E-9) #flux_x.vector(), flux_y.vector(),
		writer.writerows(everything)
  

	i += 1
	t += dt

out.close()
	
#flux = project(-mu*grad(u) + u*w, W)
#flux_x,flux_y = flux.split(deepcopy = True)

w_x,w_y = w_.split(deepcopy = True)

#output

final = open('magnetophoresis_finalT.csv', 'w')
writer = csv.writer(final, delimiter = '\t')
writer.writerow(header)
x_coord = V.tabulate_dof_coordinates()
time = [t]*len(x_coord)
everything = zip(time, x_coord[:,0], x_coord[:,1], u_.vector(), w_x.vector(), w_y.vector()) #flux_x.vector(), flux_y.vector(),  
writer.writerows(everything)
final.close()

fig = plt.figure ( )
ax = plt.subplot ( 111 )
c = plot (u_, mode='color', vmin=0, vmax=1)
ax.grid ( True )
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('Position x', fontsize = 12)
plt.ylabel('Position y', fontsize = 12)
plt.title ( 'Concentration, time %.4f' % ( t ), fontsize = 13 )
plt.colorbar(c, orientation = 'horizontal', pad = 0.2)
filename = ( 'couple_final.png' )
plt.savefig ( filename )
plt.close( )






