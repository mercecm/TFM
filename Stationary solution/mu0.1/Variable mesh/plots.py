import pandas as pd
import matplotlib.pyplot as plt

columns = ['x', 'uh', 'u_exact', 'flux', 'flux_exact']

file1 = pd.read_csv('Computed_1_grid10.csv', delimiter = '\t')
file1 = file1.sort_values(by = ['x'])

file2 = pd.read_csv('Computed_2_grid10.csv', delimiter = '\t')
file2 = file2.sort_values(by = ['x'])

file3 = pd.read_csv('Computed_1_grid100.csv', delimiter = '\t')
file3 = file3.sort_values(by = ['x'])

file4 = pd.read_csv('Computed_2_grid100.csv', delimiter = '\t')
file4 = file4.sort_values(by = ['x'])

#plot concentrations

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(file1['x'], file1['u'], label = 'Degree = 1; Grid = 10' )
plt.plot(file2['x'], file2['u'], label = 'Degree = 2; Grid = 10' )
plt.plot(file3['x'], file3['u'], label = 'Degree = 1; Grid = 100' )
plt.plot(file4['x'], file4['u'], label = 'Degree = 2; Grid = 100' )
ax.legend (fontsize = 12 )
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlim([0,1])
plt.xlabel('Position, x\'', fontsize = 12)
plt.ylim([0,1.1])
plt.ylabel('Concentration, u', fontsize = 12)
filename = ('concentrations_variable.png')
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

#  Plot also the flux of particles 

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(file1['x'], file1['f'], label = 'Degree = 1; Grid = 10' )
plt.plot(file2['x'], file2['f'], label = 'Degree = 2; Grid = 10' )
plt.plot(file3['x'], file3['f'], label = 'Degree = 1; Grid = 100' )
plt.plot(file4['x'], file4['f'], label = 'Degree = 2; Grid = 100' )
ax.legend (fontsize = 12 )
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlim([0,1])
plt.xlabel('Position, x\'', fontsize = 12)
#plt.ylim([-0.2,-1.8])
plt.ylabel('Flux, J', fontsize = 12)
filename = ( 'flux_variable.png' )
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )


