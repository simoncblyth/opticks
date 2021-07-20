"""

https://stackoverflow.com/questions/8921296/how-do-i-plot-a-step-function-with-matplotlib-in-python

New in matplotlib 3.4.0

There is a new plt.stairs method to complement plt.step:


"""
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import StepPatch

np.random.seed(0)
h, edges = np.histogram(np.random.normal(5, 3, 5000),
                        bins=np.linspace(0, 10, 20))

fig, axs = plt.subplots(3, 1, figsize=(7, 15))
axs[0].stairs(h, edges, label='Simple histogram')
axs[0].stairs(h, edges + 5, baseline=50, label='Modified baseline')
axs[0].stairs(h, edges + 10, baseline=None, label='No edges')
axs[0].set_title("Step Histograms")

axs[1].stairs(np.arange(1, 6, 1), fill=True,
              label='Filled histogram\nw/ automatic edges')
axs[1].stairs(np.arange(1, 6, 1)*0.3, np.arange(2, 8, 1),
              orientation='horizontal', hatch='//',
              label='Hatched histogram\nw/ horizontal orientation')
axs[1].set_title("Filled histogram")


if 0:
    patch = StepPatch(values=[1, 2, 3, 2, 1],
                      edges=range(1, 7),
                      label=('Patch derived underlying object\n'
                             'with default edge/facecolor behaviour'))
    axs[2].add_patch(patch)
    axs[2].set_xlim(0, 7)
    axs[2].set_ylim(-1, 5)
    axs[2].set_title("StepPatch artist")

for ax in axs:
    ax.legend()
plt.show()
