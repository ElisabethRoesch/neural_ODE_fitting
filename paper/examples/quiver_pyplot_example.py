#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np

x_coords,y_coords = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .25)) # test grid
x_grads,y_grads = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .25))   # dudt - deep neural net output
fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
q = ax.quiver(x_coords,y_coords,x_coords,y_coords)

plt.show()
