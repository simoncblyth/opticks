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

    dtype = np.float32

    OFF="T0"
    SLOW32="T1"
    SLOW16="T2" 
    SLOW8="T3" 
    SLOW4="T4" 
    SLOW2="T5" 
    NORM="T6" 
    FAST="T7"
    FAST2="T8"
    FAST4="T9"


    ####################

    n0 = 3

    eaa = np.zeros( (n0,3), dtype=dtype )
    uaa = np.zeros( (n0,3), dtype=dtype )
    caa = np.zeros( (n0,4), dtype=dtype )

    eaa[:,0] = np.linspace(-n0, -1, n0, dtype=dtype ) 
    eaa[:,1] = np.zeros( n0, dtype=dtype ) 
    eaa[:,2] = np.ones( n0, dtype=dtype )

    uaa[:n0] = [0,0,1] 

    caa[0].view("|S2")[0:8] = ["C1", "T6", "B2", "Q0", "A1", "P1", "", "" ]
    caa[2].view("|S2")[0:8] = ["Q1","","","Q0","","T5","",""] 

    #################

    pz = 1.0
    pr = 1.05

    phase0 = np.arccos(pz) 
    ta = np.linspace( 0, 2*np.pi, 32 )[:-1]
    za = np.cos(ta+phase0)
    m = np.argmin(np.abs(za[2:]-pz))+2   # index of za closest to that pz value going around again, excluding 0
    t0 = ta[:m+1]
    n1 = len(t0)
    st0 = np.sin(t0+phase0)
    ct0 = np.cos(t0+phase0)

    ebb = np.zeros( (n1,3) , dtype=dtype )  # eye position 
    ubb = np.zeros( (n1,3) , dtype=dtype ) # up direction
    cbb = np.zeros( (n1,4), dtype=dtype )  # ctrl  

    ebb[:,0] = st0
    ebb[:,1] = 0
    ebb[:,2] = ct0
    ebb *= pr

    ubb[:,0] = st0
    ubb[:,1] = 0 
    ubb[:,2] = ct0 

    cbb[0].view("|S2")[0:8] = ["C0", "","","","T6","","T7","" ] 
    cbb[1].view("|S2")[0:1] = ["T8"] 
    cbb[n1/2+0].view("|S2")[0:8] = ["E3","","","","","","","" ]
    cbb[n1/2+1].view("|S2")[0:8] = ["","","","","","","","" ]
    cbb[n1/2+2].view("|S2")[0:8] = ["","","","","","","","" ]
    cbb[n1/2+3].view("|S2")[0:8] = ["","","","","","","","" ]
    cbb[n1/2+4].view("|S2")[0:8] = ["E1","","","","","","","" ]

    cbb[n1-2].view("|S2")[0:1] = ["T5"] 

    ##############################

    # take the last point x value (close to pz) and make xy loop
    r2 = np.abs(ebb[-1,0])
    tb = np.linspace( 0, 2*np.pi, 16)[:-1]
    n2 = len(tb)

    ecc = np.zeros( (n2,3), dtype=dtype )
    ucc = np.zeros( (n2,3), dtype=dtype )
    ccc = np.zeros( (n2,4), dtype=dtype )


    ecc[:,0] = r2*np.cos(tb)
    ecc[:,1] = r2*np.sin(tb)
    ecc[:,2] = ebb[-1,2]

    ucc[:,0] = np.zeros(n2, dtype=dtype)
    ucc[:,1] = np.zeros(n2, dtype=dtype)
    ucc[:,2] = np.ones(n2, dtype=dtype)


    ccc[0].view("|S2")[0:8] = ["T7","","","" ,"","","",""]
    ccc[n2/2].view("|S2")[0:8] = ["O1","I1","","" ,"","","",""]
    ccc[n2/2+2].view("|S2")[0:8] = ["I0","O0","","" ,"","","",""]



    #########  joining together the sub-paths 

    n = n0 + n1 + n2

    eye = np.zeros( (n, 3), dtype=dtype )
    look = np.zeros( (n, 3), dtype=dtype )
    up = np.zeros( (n, 3), dtype=dtype )
    ctrl = np.zeros( (n, 4), dtype=dtype )

    eye[:n0] = eaa
    eye[n0:n0+n1] = ebb
    eye[n0+n1:n0+n1+n2] = ecc

    up[:n0] = uaa
    up[n0:n0+n1] = ubb
    up[n0+n1:n0+n1+n2] = ucc

    ctrl[:n0] = caa  
    ctrl[n0:n0+n1] = cbb
    ctrl[n0+n1:n0+n1+n2] = ccc


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
    elu[:,3,:4] = ctrl
    print(elu[:,3,:4].copy().view("|S2"))

    np.save("/tmp/flightpath.npy", elu ) 


