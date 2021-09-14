#!/usr/bin/env python
"""
steps.py
===========

::

    ipython -i steps.py 

This demonstrates that should use drawstyle="steps-post"
when plotting rindex as that appears to duplicate the 
last value to give same number of "edges" to "values".
"""
import numpy as np
import matplotlib.pyplot as plt

edges = np.array([0,1,2,3])
values_ = np.array([10,20,30])   

dupe_last_value = True 
if dupe_last_value:
    values = np.zeros( len(values_) + 1 ) 
    values[:-1] = values_
    values[-1] = values_[-1]
else:
    values = values_
pass

fig,axs = plt.subplots(2,2, figsize=[12.8, 7.2])
ds=[["steps", "steps-pre"],["steps-mid", "steps-post"] ]

title = " edges : %s   values : %s  dupe_last_value:%s  " % (repr(edges), repr(values), "Y" if dupe_last_value else "N")
print(title)
fig.suptitle( title )

ylim = [ 0 , values[-1]+10 ]
xlim = [ edges[0]-1, edges[-1]+1 ]


for ix in [0,1]:
    for iy in [0,1]:
        ax = axs[ix,iy]
        drawstyle = ds[ix][iy]
       
        ax.plot( edges, values, drawstyle=drawstyle, label=drawstyle )
        ax.scatter( edges, values )

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.legend()

        for e in edges:
            ax.plot( [e, e], ylim, linestyle="dotted", color="r")
        pass 
        for v in values:
            ax.plot( xlim, [v,v], linestyle="dotted", color="r")
        pass 
    pass
pass

fig.show()  



