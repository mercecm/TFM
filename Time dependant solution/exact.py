import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

t0 = 0.0001
t1 = 0.0020
t2 = 0.0040


L = 0.05
w = 0.16E-6
D = 4.3E-11
mu = D/(L*w)

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

x1 = 0.005
x2 = 0.03 
xe = np.linspace(0.0,1.0,1000)
xe = xe.astype(float)
te = np.linspace(0.0,0.05,100)

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(te,flux_surf(te,mu), label = 'x = 0.0')
plt.plot(te,flux_exact(x1,te,mu), label = 'x = %.3f' %x1)
plt.plot(te,flux_exact(x2,te,mu), label = 'x = %.2f' %x2)
ax.legend ( )
plt.xlabel('Time')
plt.ylabel('Flux')
ax.grid ( True )
plt.title ( 'Exact flux in surface' )
filename = ( 'flux_exact_3surf.png' )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
plt.close ( )

#fig = plt.figure ( )
#ax = plt.subplot ( 111 )
#plt.plot(te,flux_exact(x1,te,mu))
#ax.legend ( )
#ax.grid ( True )
#plt.title ( 'Flux in surface x=%.2f' %x1 )
#filename = ( 'flux_surf_x1.png' )
#plt.savefig ( filename )
#print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
#plt.close ( )

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(xe,flux_exact(xe,t0,mu), label = 't = %f' %t0)
plt.plot(xe,flux_exact(xe,t1,mu), label = 't = %f' %t1)
plt.plot(xe,flux_exact(xe,t2,mu), label = 't = %f' %t2)
plt.plot(xe,flux_stationary(xe,mu), label = 'stationary solution')
ax.legend ( )
plt.xlabel('Position')
plt.ylabel('Flux')
ax.grid ( True )
plt.title ( 'Exact solution for flux' )
filename = ( 'flux_exact.png' )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
plt.close ( )

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(xe,c_exact(xe,t0,mu), label = 't = %f' %t0)
plt.plot(xe,c_exact(xe,t1,mu), label = 't = %f' %t1)
plt.plot(xe,c_exact(xe,t2,mu), label = 't = %f' %t2)
plt.plot(xe,c_stationary(xe,mu), label = 'stationary solution')
ax.legend ( )
plt.xlabel('Position')
plt.ylabel('Concentration')
ax.grid ( True )
plt.title ( 'Exact solution for concentration' )
filename = ( 'concentration_exact.png' )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
plt.close ( )
