from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

def z(u,v):
    return 10 * u*v

u, v = np.neshgrid(x,y)
fig = plt.figure()
ax = plt.gca(projection='3d')
ax.plot_surface(u, v, z(u,v), cmap = plt.get_cmap("jet"))

plt.show()
