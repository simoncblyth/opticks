simtrace-shakedown
======================


After run and grab with new sframe based machinery::

    epsilon:CSGOptiX blyth$ ./cxs_Hama.sh ana


* analysis giving blank, with just genstep points. unsurprisingly. 
* range of plotting window too small, gridscale extent ?


Checking with SIM plotting shows a big ring so it appears
are not actually starting the rays from the desired points 
on the genstep grid.::  

    SIM=1 ./cxs_Hama.sh ana

::


In [2]: t.photon[:,0,3]                                                                                                                                                                                   
Out[2]: array([17699.006, 17700.979, 17701.773, 17701.395, 17701.018, ..., 17699.848, 17698.197, 17699.15 , 17701.758, 17698.188], dtype=float32)

In [3]: t.photon[:,0,3].min()                                                                                                                                                                             
Out[3]: 17698.164

In [4]: t.photon[:,0,3].max()                                                                                                                                                                             
Out[4]: 17829.307





::

    203 /**
    204 qevent::add_simtrace
    205 ----------------------
    206 
    207 NB simtrace "photon" *a* is very different from real ones
    208 
    209 a.q0.f
    210     prd.q0.f normal, distance, aka "isect" 
    211 
    212 a.q1
    213     intersect position from pos+t*dir, 0.
    214 
    215 a.q2
    216     initial pos, tmin
    217 
    218 a.q3 
    219     initial dir, prd.identity
    220 
    221 
    222 **/
    223 
    224 QEVENT_METHOD void qevent::add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin )
    225 {
    226     float t = prd->distance() ; 
    227     quad4 a ;
    228     
    229     a.q0.f  = prd->q0.f ;
    230     
    231     a.q1.f.x = p.q0.f.x + t*p.q1.f.x ;
    232     a.q1.f.y = p.q0.f.y + t*p.q1.f.y ;
    233     a.q1.f.z = p.q0.f.z + t*p.q1.f.z ;
    234     a.q1.i.w = 0.f ;  
    235     
    236     a.q2.f.x = p.q0.f.x ;
    237     a.q2.f.y = p.q0.f.y ;
    238     a.q2.f.z = p.q0.f.z ;
    239     a.q2.u.w = tmin ; 
    240     
    241     a.q3.f.x = p.q1.f.x ;
    242     a.q3.f.y = p.q1.f.y ;
    243     a.q3.f.z = p.q1.f.z ;
    244     a.q3.u.w = prd->identity() ;
    245     
    246     const sphoton& s = (sphoton&)a ;
    247     photon[idx] = s ;
    248 }   





::

    In [1]: t.photon                                                                                                                                                                                          
    Out[1]: 
    array([[[    -0.062,      0.   ,     -0.998,  17699.006],
            [ -1088.579,      0.   , -17666.494,      0.   ],
            [    -1.6  ,      0.   ,     -0.9  ,      0.   ],
            [    -0.061,      0.   ,     -0.998,      0.   ]],

           [[     0.879,      0.   ,     -0.476,  17700.979],
            [ 15562.83 ,      0.   ,  -8431.387,      0.   ],
            [    -1.6  ,      0.   ,     -0.9  ,      0.   ],
            [     0.879,      0.   ,     -0.476,      0.   ]],

           [[     0.97 ,      0.   ,      0.243,  17701.773],
            [ 17170.807,      0.   ,   4295.747,      0.   ],
            [    -1.6  ,      0.   ,     -0.9  ,      0.   ],
            [     0.97 ,      0.   ,      0.243,      0.   ]],

           [[     0.981,      0.   ,     -0.194,  17701.395],
            [ 17364.271,      0.   ,  -3431.041,      0.   ],
            [    -1.6  ,      0.   ,     -0.9  ,      0.   ],
            [     0.981,      0.   ,     -0.194,      0.   ]],

           [[     0.891,      0.   ,     -0.453,  17701.018],
            [ 15777.213,      0.   ,  -8023.06 ,      0.   ],
            [    -1.6  ,      0.   ,     -0.9  ,      0.   ],
            [     0.891,      0.   ,     -0.453,      0.   ]],

           ...,

           [[     0.562,      0.   ,     -0.827,  17699.848],
            [  9945.679,      0.   , -14641.499,      0.   ],
            [     1.6  ,      0.   ,      0.9  ,      0.   ],
            [     0.562,      0.   ,     -0.827,      0.   ]],

           [[     0.947,      0.   ,      0.32 ,  17698.197],
            [ 16769.418,      0.   ,   5663.622,      0.   ],
            [     1.6  ,      0.   ,      0.9  ,      0.   ],
            [     0.947,      0.   ,      0.32 ,      0.   ]],

           [[    -0.029,      0.   ,      1.   ,  17699.15 ],
            [  -520.058,      0.   ,  17692.361,      0.   ],
            [     1.6  ,      0.   ,      0.9  ,      0.   ],
            [    -0.029,      0.   ,      1.   ,      0.   ]],

           [[    -0.976,      0.   ,     -0.217,  17701.758],
            [-17279.29 ,      0.   ,  -3836.175,      0.   ],
            [     1.6  ,      0.   ,      0.9  ,      0.   ],
            [    -0.976,      0.   ,     -0.217,      0.   ]],

           [[     0.936,      0.   ,      0.352,  17698.188],
            [ 16565.639,      0.   ,   6234.555,      0.   ],
            [     1.6  ,      0.   ,      0.9  ,      0.   ],
            [     0.936,      0.   ,      0.352,      0.   ]]], dtype=float32)




Initial pos is in a grid, but very small one around origin::

    In [5]: t.photon[:,2]                                                                                                                                                                                     
    Out[5]: 
    array([[-1.6,  0. , -0.9,  0. ],
           [-1.6,  0. , -0.9,  0. ],
           [-1.6,  0. , -0.9,  0. ],
           [-1.6,  0. , -0.9,  0. ],
           [-1.6,  0. , -0.9,  0. ],
           ...,
           [ 1.6,  0. ,  0.9,  0. ],
           [ 1.6,  0. ,  0.9,  0. ],
           [ 1.6,  0. ,  0.9,  0. ],
           [ 1.6,  0. ,  0.9,  0. ],
           [ 1.6,  0. ,  0.9,  0. ]], dtype=float32)

    In [6]: t.photon[:,2].shape                                                                                                                                                                               
    Out[6]: (313500, 4)

::

    mom =  t.photon[:,3,:3]   

    In [15]: np.sum( mom*mom, axis=1 )                                                                                                                                                                        
    Out[15]: array([1., 1., 1., 1., 1., ..., 1., 1., 1., 1., 1.], dtype=float32)

    In [16]: np.sum( mom*mom, axis=1 ).min()                                                                                                                                                                  
    Out[16]: 0.9999995

    In [17]: np.sum( mom*mom, axis=1 ).max()                                                                                                                                                                  
    Out[17]: 1.0000002





::

    In [20]: frame                                                                                                                                                                                            
    Out[20]: 
    sframe     : 
    path       : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGOptiXSimtraceTest/sframe.npy
    meta       : {'creator': 'sframe::save', 'frs': 'Hama'}
    ce         : array([0., 0., 0., 0.], dtype=float32)
    grid       : ix0  -16 ix1   16 iy0    0 iy1    0 iz0   -9 iz1    9 num_photon  500 gridscale     0.1000
    target     : midx      0 mord      0 iidx      0       inst       0   
    qat4id     : ins_idx      0 gas_idx    0    0 
    m2w        : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    w2m        : 
    array([[ 1., -0.,  0.,  0.],
           [-0.,  1., -0.,  0.],
           [ 0., -0.,  1.,  0.],
           [-0.,  0., -0.,  1.]], dtype=float32)

    id         : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)


Genstep transforms are just local shifts::

    In [24]: t.genstep[:,2:]                                                                                                                                                                                  
    Out[24]: 
    array([[[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [-1.6,  0. , -0.9,  1. ]],

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [-1.6,  0. , -0.8,  1. ]],

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [-1.6,  0. , -0.7,  1. ]],

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [-1.6,  0. , -0.6,  1. ]],

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [-1.6,  0. , -0.5,  1. ]],

           ...,

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [ 1.6,  0. ,  0.5,  1. ]],

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [ 1.6,  0. ,  0.6,  1. ]],

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [ 1.6,  0. ,  0.7,  1. ]],

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [ 1.6,  0. ,  0.8,  1. ]],

           [[ 1. ,  0. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ,  0. ],
            [ 0. ,  0. ,  1. ,  0. ],
            [ 1.6,  0. ,  0.9,  1. ]]], dtype=float32)


