import pandas as pd
import matplotlib.pyplot as plt

file1 = pd.read_csv('Computed_mu0.005.csv', delimiter = '\t')
file1 = file1.sort_values(by = ['x'])

#plot concentrations

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(file1['x'], file1['u1'], label = 't = 0.0001' )
plt.plot(file1['x'], file1['u2'], label = 't = 0.005' )
plt.plot(file1['x'], file1['u'], label = 't = 0.05' )
ax.legend (fontsize = 12 )
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('Position, x\'', fontsize = 12)
plt.ylabel('Concentration, u', fontsize = 12)
plt.title ( 'Computed concentration, grid 50' )

a = plt.axes([0.3, 0.2, .2, .5])
plt.plot(file1['x'], file1['u1'])
plt.plot(file1['x'], file1['u2'])
plt.plot(file1['x'], file1['u'])
#plt.title('Impulse response')
plt.xlim(0, 0.05)
plt.xticks()
plt.yticks()

filename = ('concentrations_3time.png')
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

#  Plot also the flux of particles 

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(file1['x'], file1['f1'], label = 't = 0.0001' )
plt.plot(file1['x'], file1['f2'], label = 't = 0.005' )
plt.plot(file1['x'], file1['f'], label = 't = 0.05' )
ax.legend (fontsize = 12 )
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('Position, x\'', fontsize = 12)
plt.ylabel('Flux, J', fontsize = 12)
plt.title ( 'Computed flux, grid 50' )

a = plt.axes([0.3, 0.2, .15, .5])
plt.plot(file1['x'], file1['f1'])
plt.plot(file1['x'], file1['f2'])
plt.plot(file1['x'], file1['f'])
plt.xlim(0, 0.05)
plt.xticks()
plt.yticks()

filename = ('flux_3time.png')
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

