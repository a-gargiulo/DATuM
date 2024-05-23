"""
Insert docstring...
"""
from ..datum.beverli import Beverli

# Create a hill of type Beverli
hill = Beverli()

# Visualize hill surface (interactively, in browser window)
# and x1-x2 profiles in 3D
fig = hill.plotter.plot_3d_hill()
fig = hill.plotter.plot_x1_x2_profile_3d(0, fig)
fig = hill.plotter.plot_x1_x2_profile_3d(0.15, fig)
fig = hill.plotter.plot_x1_x2_profile_3d(0.3, fig)
fig.show()
