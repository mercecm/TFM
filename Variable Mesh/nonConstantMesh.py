from dolfin import *
from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
 # https://fenicsproject.org/qa/13422/refining-mesh-near-boundaries/
 
def create_mesh(nx,ny):
  m = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), nx, ny)
  x = m.coordinates()
  #In order to have a variable mesh one needs to apply a transformation to x. To redefine all the coordinates, x[:]; to redefine only one coordinate, x[:,1] or x[:,0]
  x[:,0] = x[:,0]**2
  return m

a = 4
b = 8
mesh = create_mesh(a,b)
plot (mesh)
filename = ('Mesh2-%d x %d' %(a,b))
plt.title('Transformation y=x**2')
plt.savefig(filename)
plt.show()

