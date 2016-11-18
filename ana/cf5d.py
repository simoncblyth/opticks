"""

5d comparison (ie staying in blib form) is tedious, m
instead moved to trasforming the blib into mlib form
and comparing that over in bnd.py 


"""


def cf5d(b0,i1,i2):

    KL = odict()
    KL["RINDEX"]   = (0,PropLib.M_REFRACTIVE_INDEX)
    KL["ABSLEN"]   = (0,PropLib.M_ABSORPTION_LENGTH)
    KL["RAYLEIGH"] = (0,PropLib.M_SCATTERING_LENGTH)
    KL["REEMPROB"] = (0,PropLib.M_REEMISSION_PROB)
    KL["GROUPVEL"] = (1,PropLib.L_GROUP_VELOCITY)

    shape = b0.data.shape

    ni = shape[0]   # bnd
    nj = shape[1]   # omat/osur/isur/imat
    nk = shape[2]   # g0,g1
    nl = shape[3]   # samples = 39 for identity, or 761 for interpolating 
    nm = shape[4]   # props   

    cf = np.zeros( (ni, nj, nk, nm, 2), dtype=np.float32  )

    i12 = i1.data - i2.data
    #np.where( i12 > 300)      difficult to interpret the 5-tuple of indices

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):

                b1.dat.ijk = i,j,k
                b2.dat.ijk = i,j,k
                i1.dat.ijk = i,j,k
                i2.dat.ijk = i,j,k

                assert np.allclose( b1.dat.d, b1.dat.d )

                for m in range(nm):
                    i12 = i1.dat.d[:,m] - i2.dat.d[:,m]
                    cf[i,j,k,m,0] = i12.min()
                    cf[i,j,k,m,1] = i12.max()
                pass
            pass
        pass
    pass

    # big discreps in all Water flavors  absorption length
    for i in range(ni):
        for j in range(nj):
            if cf[i,j].min() < -1 or cf[i,j].max() > 1:
                 print i, j, b0.names[i],"\n", cf[i,j]



    cfm = odict()
    
    for mat in b0.bnd.mats:
        oms = b0.bnd.oms(mat)   # indices of omat
        ims = b0.bnd.ims(mat)

        print mat
        for om in oms:
            cfom = cf[om,0]
            if mat in cfm:
                assert np.all(cfm[mat] == cfom)
            else:
                cfm[mat] = cfom 
            pass
            #if cfom.min() < -1 or cfom.max() > 1:
            #     print "om", b0.names[om],"\n", cfom

        for im in ims:

            cfim = cf[im,3]
            if mat in cfm:
                assert np.all(cfm[mat] == cfim)
            else:
                cfm[mat] = cfim
            pass
            #if cfim.min() < -1 or cfim.max() > 1:
            #     print "im", b0.names[im],"\n", cfim


    # interpol discreps seem big in absolute terms
    # but maybe not in relative 

if 0:

    for mat in cfm.keys():
        if cfm[mat].min() < -0.001 or cfm[mat].max() > 0.001:
            print mat,"\n", cfm[mat].reshape(-1,2).T

            oms = b0.bnd.oms(mat)
            if len(oms) > 0:
                i1.dat.ijk = oms[0],0,0
                i2.dat.ijk = oms[0],0,0
                for m in range(nm):
                    print m,"\n",(i2.dat.d[:,m] - i1.dat.d[:,m]) / (i2.dat.d[:,m] + i1.dat.d[:,m])/2.

                i1.dat.ijk = oms[0],0,1
                i2.dat.ijk = oms[0],0,1
                for m in range(nm):
                    print m,"\n",(i2.dat.d[:,m] - i1.dat.d[:,m]) / (i2.dat.d[:,m] + i1.dat.d[:,m])/2.





