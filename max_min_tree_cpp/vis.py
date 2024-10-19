import cv2
import matplotlib as mpl
import numpy as np

from matplotlib import ticker
from matplotlib import pyplot as plt
from pathlib import Path

path = Path("assets/ACDC_ORIGIN/1.png")

data = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))[100:250, 100:250]

cv2.imwrite("assets/part/100-250.png", data)

x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

# # ax = fig.add_subplot(subplot_kw={"projection": "3d"})
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# fig = plt.figure(figsize=(10, 20))
# ax = plt.axes(projection='3d')

# ax.tick_params('both', length=20, width=5)
# ax.set_axes_locator() .set_axes_locator(ticker.MultipleLocator(10))

# Plot the surface.
surf = ax.plot_surface(x, y, data, cmap='viridis',
                       linewidth=0, antialiased=False)

plt.show()
pass