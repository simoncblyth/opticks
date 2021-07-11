#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


r = 17820
c = plt.Circle((0, 0), r, fill=False, ec='red' )

a = 450
s = r - np.sqrt( np.power(r,2) - np.power(a,2) )

b = plt.Rectangle( (-a,r-s), 2*a, 2*a, fill=False, ec='blue' )

bbox = plt.Rectangle( (-r,-r),    2*r, 2*r+2*a-s , fill=False, ec='blue' )


fig, ax = plt.subplots() 

ax.set_xlim( -r*1.1, r*1.1 )
ax.set_ylim( -r*1.1, r*1.1 )
ax.set_aspect('equal')


ax.add_patch(b)
ax.add_patch(c)
ax.add_patch(bbox)

ax.plot( [0, 0], [0, r*1.1], linestyle="dashed" )


fig.show()



#fig.savefig('/tmp/sagitta.png')

