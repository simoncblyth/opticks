#!/usr/bin/env python

import numpy as np
import ciexyz.ciexyz as c

wf = np.linspace(300,800,501, dtype=np.float32)
wd = np.linspace(300,800,501)

xf = c.X(wf)
yf = c.Y(wf)
zf = c.Z(wf)
bb5kf = c.BB5K(wf)
bb6kf = c.BB6K(wf)

xd = c.X(wd)
yd = c.Y(wd)
zd = c.Z(wd)
bb5kd = c.BB5K(wd)
bb6kd = c.BB6K(wd)

xyz = np.empty([len(wd),3])
xyz[:,0] = xd
xyz[:,1] = yd
xyz[:,2] = zd

print xyz

for i, c in enumerate([xf,xd,yf,yd,zf,zd,bb5kf,bb5kd,bb6kf,bb6kd]):
    print "%2d min/max  %s %s  " % ( i, c.min(), c.max() )

print bb5kd
print bb6kd


