#!/usr/bin/env python
"""
Hmm need to make connection to the volume traversal index 
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.geocache import keydir
from opticks.ana.prim import Dir
from opticks.ana.geom2d import Geom2d

np.set_printoptions(suppress=True)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)
    log.info(kd)
    assert os.path.exists(kd), kd 

    os.environ["IDPATH"] = kd    ## TODO: avoid having to do this, due to prim internals

    mm0 = Geom2d(kd, ridx=0)

    target = 352854   # guide tube torus, is convenient frame 
    sc = mm0.ce[target][3]/1000.   #  torus big radius in meters   17.838   


    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D 
    import mpl_toolkits.mplot3d.art3d as art3d

    plt.ion()
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111,projection='3d')
    plt.title("mm0 geom2d")
    sz = 25

    ax.set_xlim([-sz,sz])
    ax.set_ylim([-sz,sz])
    ax.set_zlim([-sz,sz])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    mm0.render(ax, art3d=art3d)

    pz = 1.0
    pr = 1.1

    dtype = np.float32

    phase0 = np.arccos(pz) 
    ta = np.linspace( 0, 2*np.pi, 32 )[:-1]
    za = np.cos(ta+phase0)
    m = np.argmin(np.abs(za[2:]-pz))+2   # index of za closest to that pz value going around again, excluding 0
    t0 = ta[:m+1]
    m0 = len(t0)
    st0 = np.sin(t0+phase0)
    ct0 = np.cos(t0+phase0)

    oxz = np.zeros( (m0,3) , dtype=dtype )
    oxz[:,0] = st0
    oxz[:,1] = 0
    oxz[:,2] = ct0
    oxz *= pr


    uxz = np.zeros( (m0,3) , dtype=dtype )
    uxz[:,0] = st0
    uxz[:,1] = 0 
    uxz[:,2] = ct0 


    # take the last point x value (close to pz) and make xy loop
    r2 = np.abs(oxz[-1,0])
    tb = np.linspace( 0, 2*np.pi, 8)[:-1]
    m1 = len(tb)

    oxy = np.zeros( (m1,3), dtype=dtype )
    oxy[:,0] = r2*np.cos(tb)
    oxy[:,1] = r2*np.sin(tb)
    oxy[:,2] = oxz[-1,2]

    uxy = np.zeros( (m1,3), dtype=dtype )
    uxy[:,0] = np.zeros(m1, dtype=dtype)
    uxy[:,1] = np.zeros(m1, dtype=dtype)
    uxy[:,2] = np.ones(m1, dtype=dtype)

    n = 1 + m0 + m1
    eye = np.zeros( (n, 3), dtype=dtype )
    look = np.zeros( (n, 3), dtype=dtype )
    up = np.zeros( (n, 3), dtype=dtype )
    ctrl = np.zeros( (n, 4), dtype=dtype )

    eye[0] = [-1, 0, 0] 
    eye[1:1+m0] = oxz
    eye[1+m0:1+m0+m1] = oxy

    up[0] = [0,0,1]
    up[1] = [0,0,1]
    up[1:1+m0] = uxz
    up[1+m0:1+m0+m1] = uxy



    ## test ctrl message, see SCtrl
    mid = ctrl[m0/2].view("|S2")
    mid[0] = "c0" ; 
    mid[1] = "c1" ; 
    mid[2] = "aa" ; 



    look[:-1] = eye[1:]
    look[-1] = eye[0]

    gaze = look - eye

    x = sc*eye[:,0] 
    y = sc*eye[:,1] 
    z = sc*eye[:,2]

    u0 = gaze[:, 0] 
    v0 = gaze[:, 1] 
    w0 = gaze[:, 2] 

    u1 = up[:, 0] 
    v1 = up[:, 1] 
    w1 = up[:, 2] 

   
    #ax.plot( x,z )
    ax.quiver( x, y, z, u0, v0, w0  ) 
    ax.quiver( x, y, z, u1, v1, w1  ) 



    labels = False
    if labels:
        for i in range(len(eye)):
            plt.text( x[i], y[i], z[i], i , "z" )
        pass  
    pass


    fig.show()

    elu = np.zeros( (n,4,4), dtype=np.float32)
    elu[:,0,:3] = eye 
    elu[:,1,:3] = look
    elu[:,2,:3] = up
    elu[:,3] = ctrl

    # print elu[:,3].copy().view("|S2")

    np.save("/tmp/flightpath.npy", elu ) 


