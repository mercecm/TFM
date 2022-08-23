import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

t0 = 0.0001
t1 = 0.0025
t2 = 0.0050


def c_exact(x,t,mu):
  c = (1-np.exp(-x/mu)) - (1/2)*(erfc((x+t)/(np.sqrt(4*mu*t))) - np.exp(-x/mu)*(2-erfc((x-t)/(np.sqrt(4*mu*t)))))
  return c

def c_stationary(x,mu):
  c = (1-np.exp(-x/mu))
  return c

def c_stationary2(x,mu):
  c = (np.exp(-x/mu)-1)/(np.exp(-1/mu)-1)
  return c
  
def flux_exact(x,t,mu):
  f_e = -1
  f_t = (1/2)*(-np.sqrt(mu/(np.pi*t))*np.exp(-(x+t)*(x+t)/(4*mu*t)) - np.sqrt(mu/(np.pi*t))*np.exp(-x/mu)*np.exp(-(x-t)*(x-t)/(4*mu*t)) + erfc((x+t)/np.sqrt(4*mu*t)))
  f = f_e + f_t
  return f

def flux_stationary(x,mu):
  f = -np.exp(x-x)
  return f

def flux_surf(t,mu):
  f = 1-(1/2)*erfc(np.sqrt(t/(4*mu))) + np.sqrt(mu/(np.pi*t))*np.exp(-t/(4*mu))
  return f

#x1 = 0.005
#x2 = 0.03
x1 = 0.05
x2 = 0.3
xe = np.linspace(0.0,1.0,10000)
xe = xe.astype(float)
te = np.linspace(0.0,0.75,1000)

fig = plt.figure ( )
ax = plt.subplot ( 111 )
for mu in [0.01, 0.05, 0.1]:
  plt.plot(xe,c_stationary2(xe,mu), label = 'mu = %.2f' %mu)
ax.legend ( )
plt.xlabel('Position, x\'')
plt.ylabel('Concentration, u')
plt.title ( 'Exact solution for concentration' )
filename = ( 'stationary_analytic_mu.png' )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
plt.close ( )

L = 0.05#/50
#L = 0.005
w = 0.16E-6
D = 4.3E-11
mu = D/(L*w)
#mu = 0.1

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(te,flux_surf(te,mu), label = 'x = 0.0')
plt.plot(te,-flux_exact(x1,te,mu), label = 'x = %.3f' %x1)
plt.plot(te,-flux_exact(x2,te,mu), label = 'x = %.2f' %x2)
ax.legend ( )
plt.xlabel('Time, t\'')
plt.ylabel('Flux, J')
plt.ylim(0,14)
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
plt.plot(xe,flux_exact(xe,t0,mu), label = 't = %.4f' %t0)
#plt.plot(xe,flux_exact(xe,t1,mu), label = 't = %.4f' %t1)
plt.plot(xe,flux_exact(xe,t2,mu), label = 't = %.3f' %t2)
plt.plot(xe,flux_stationary(xe,mu), label = 'stationary solution')
ax.legend ( )
plt.xlabel('Position, x\'')
plt.ylabel('Flux, J')
plt.title ( 'Exact solution for flux' )

a = plt.axes([0.3, 0.2, .15, .5])
plt.plot(xe,flux_exact(xe,t0,mu))
plt.plot(xe,flux_exact(xe,t2,mu))
plt.plot(xe,flux_stationary(xe,mu))
plt.xlim(0, 0.05)
plt.xticks()
plt.yticks()


filename = ( 'flux_exact.png' )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
plt.close ( )

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(xe,c_exact(xe,t0,mu), label = 't = %.4f' %t0)
#plt.plot(xe,c_exact(xe,t1,mu), label = 't = %.4f' %t1)
plt.plot(xe,c_exact(xe,t2,mu), label = 't = %.3f' %t2)
#plt.plot(xe,c_exact(xe,0.75,mu), label = 't = 0.75')
plt.plot(xe,c_stationary(xe,mu), label = 'stationary solution')
ax.legend ( )
plt.xlabel('Position, x\'')
plt.ylabel('Concentration, u')
plt.title ( 'Exact solution for concentration' )

a = plt.axes([0.3, 0.2, .2, .5])
plt.plot(xe,c_exact(xe,t0,mu))
plt.plot(xe,c_exact(xe,t2,mu))
plt.plot(xe,c_stationary(xe,mu))
plt.xlim(0, 0.05)
plt.xticks()
plt.yticks()

filename = ( 'concentration_exact.png' )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
#plt.show()
plt.close ( )
