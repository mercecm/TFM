import pandas as pd
import matplotlib.pyplot as plt

file1 = pd.read_csv('CDeqPBC_50_x_50.csv', delimiter = '\t')
file1 = file1.sort_values(by = ['x'])


surfx = file1.loc[file1['y'] == 0.25]
surfx = surfx.sort_values(by = ['x'])
surfy = file1.loc[file1['x'] == 0.5042]
surfy = surfy.sort_values(by = ['y'])

#plot concentrations

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(surfx['x'], surfx['u'])
plt.xlabel('x\' coordinate', fontsize = 12)
plt.ylabel('Concentration, u', fontsize = 12)
plt.title ( 'Computed concentration in the surface y\' = 0.25' )
filename = ('concentrations_x.png')
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )

#  Plot also the flux of particles 

fig = plt.figure ( )
ax = plt.subplot ( 111 )
plt.plot(surfy['y'], surfy['u'])
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('y\' coordinate', fontsize = 12)
plt.ylabel('Concentration, u', fontsize = 12)
plt.ylim([-0.05,1.05])
plt.title ( 'Computed concentration in the surface x\' = 0.5', fontsize = 13)
filename = ('concentrations_y.png')
plt.savefig ( filename )
print ( '  Graphics saved as "%s"' % ( filename ) )
plt.close ( )
