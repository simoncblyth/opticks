GeneralSphereDEV
===================



Curious no isect from genstep IXYZ (0,0,0)
---------------------------------------------

::


    In [15]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == 0) ]                                                                                
    Out[15]: array([], shape=(0, 4), dtype=int8)

    In [16]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == 1) ]                                                                                
    Out[16]: 
    array([[ 0,  0,  1,  3],
           [ 0,  0,  1,  5],
           [ 0,  0,  1, 11],
           [ 0,  0,  1, 14],
           [ 0,  0,  1, 20],
           [ 0,  0,  1, 33],
           [ 0,  0,  1, 43],
           [ 0,  0,  1, 52],
           [ 0,  0,  1, 53],
           [ 0,  0,  1, 54],
           [ 0,  0,  1, 57],
           [ 0,  0,  1, 87]], dtype=int8)

    In [17]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == -1) ]                                                                               
    Out[17]: 
    array([[ 0,  0, -1,  4],
           [ 0,  0, -1, 16],
           [ 0,  0, -1, 32],
           [ 0,  0, -1, 46],
           [ 0,  0, -1, 54],
           [ 0,  0, -1, 56],
           [ 0,  0, -1, 57],
           [ 0,  0, -1, 77],
           [ 0,  0, -1, 86],
           [ 0,  0, -1, 99]], dtype=int8)



