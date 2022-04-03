import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

#mu = 0.01
#cb = 1.0
t0 = 0.0001
t1 = 0.0020
t2 = 0.0040
t3 = 1.0 #estacionari

L = 0.05
w = 0.16E-6
D = 4.3E-11
mu = D/(L*w)

x = np.linspace(0,1,1000)

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

t = np.linspace(0,0.05,100)
fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(t,flux_surf(t,mu))
ax.legend ( )
ax.grid ( True )
plt.title ( 'Flux in surface x=0' )
filename = ( 'flux_surf.png' )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.show()
#plt.close ( )

#fig = plt.figure ( )
#ax = plt.subplot ( 111 )
#plt.plot(x,flux_exact(x,t0,mu), label = 't = %f' %t0)
#plt.plot(x,flux_exact(x,t1,mu), label = 't = %f' %t1)
#plt.plot(x,flux_exact(x,t2,mu), label = 't = %f' %t2)
#plt.plot(x,flux_stationary(x,mu), label = 'stationary solution')
#ax.legend ( )
#ax.grid ( True )
#plt.title ( 'exact solution at time %f and %f' % ( t1,t2 ) )
#filename = ( 'flux_exact_t%f.png' % ( t1 ) )
#plt.savefig ( filename )
#print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
#plt.close ( )
