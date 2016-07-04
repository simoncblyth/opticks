
def check_radius(sc, sli=slice(None)):
    """
    """
    s_p1 = sc.ssim.recpost(1)[:,:3]
    p_p1 = sc.psim.recpost(1)[:,:3]

    rs = np.linalg.norm(s_p1, 2, 1)
    rp = np.linalg.norm(p_p1, 2, 1)

    log.info("rs %s %s " % (rs.min(), rs.max()))
    log.info("rp %s %s " % (rp.min(), rp.max()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(s_p1[sli,0], s_p1[sli,1], s_p1[sli,2] )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.auto_scale_xyz( [-100,100], [-100,100], [-100,100] )



def check_intersect(evt, radius=100):
    """
    Assuming incident direction along +X axis
    """
    pos = evt.rpost_(0)[:,:3]

    yz = np.clip( np.linalg.norm(pos[:,1:], 2, 1), 0, radius ) 
    x  = np.sqrt(radius*radius - yz*yz )

    isp = np.copy(pos)   # point of first intersection with sphere
    isp[:,0] = -x 
    
    return isp 


def check_polarization(evt):
    """

    ::   

        In [50]: p1[~msk].shape
        Out[50]: (574, 3)

        In [51]: p1[msk].shape
        Out[51]: (999426, 3) 

    """
    p1 = evt.rpost_(1)[:,:3]
    rp1 = np.linalg.norm(p1,2,1)
    msk = np.abs( rp1 - 100. ) < 0.1

    o1 = evt.rpol_(1)[:,:3]


