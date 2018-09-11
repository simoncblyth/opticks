ckm_cerenkov_generation_align_g4_ok_deviations
=================================================


::

    epsilon:CerenkovMinimal blyth$ ckm-;ckm-so
    # source/evt/g4live/natural/-1/so.npy source/evt/g4live/natural/1/ox.npy
    import numpy as np, commands

    apath = "source/evt/g4live/natural/-1/so.npy"
    bpath = "source/evt/g4live/natural/1/ox.npy"

    print " ckm-xx- comparing so.npy and ox.npy between two dirs " 

    print "  ", commands.getoutput("date")
    print "a ", commands.getoutput("ls -l %s" % apath)
    print "b ", commands.getoutput("ls -l %s" % bpath)

    a = np.load(apath)
    b = np.load(bpath)

    print "a %s " % repr(a.shape)
    print "b %s " % repr(b.shape)

    dv = np.max( np.abs(a[:,:3]-b[:,:3]), axis=(1,2) )

    print "max deviation %s " % dv.max() 

    cuts = [1e-5, 1e-6, 1e-7, 1e-8]
    for cut in cuts:
        wh = np.where( dv > cut )[0] 
        print " deviations above cut %s num_wh %d" % ( cut, len(wh) )
        for i in wh[:10]:
            print i, dv[i], "\n",np.hstack([a[i,:3],(a[i,:3]-b[i,:3])/cut,b[i,:3]])
        pass
    pass


    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/ckm/ckm-so.py
     ckm-xx- comparing so.npy and ox.npy between two dirs 
       Sat Sep  8 20:36:17 CST 2018
    a  -rw-r--r--  1 blyth  staff  14224 Sep  8 20:36 source/evt/g4live/natural/-1/so.npy
    b  -rw-r--r--  1 blyth  staff  14224 Sep  8 20:36 source/evt/g4live/natural/1/ox.npy
    a (221, 4, 4) 
    b (221, 4, 4) 
    max deviation 6.1035156e-05 
     deviations above cut 1e-05 num_wh 54
    7 3.0517578e-05 
    [[  0.3326  -0.0666  -0.0134   0.0013   0.       0.0007   0.       0.       0.3326  -0.0666  -0.0134   0.0013]
     [  0.7503  -0.2236  -0.6221   1.       0.602   -0.0238   0.7391   0.       0.7503  -0.2236  -0.6221   1.    ]
     [ -0.6331   0.028   -0.7736 342.7441   0.7153  -0.2107  -0.5901   3.0518  -0.6331   0.028   -0.7736 342.7441]]
    13 1.5258789e-05 
    [[  0.1081  -0.0217  -0.0044   0.0004   0.0007   0.       0.       0.       0.1081  -0.0217  -0.0044   0.0004]
     [  0.8161   0.1018  -0.5688   1.       0.       0.0075  -0.006    0.       0.8161   0.1018  -0.5688   1.    ]
     [ -0.5444   0.4655  -0.6978 147.9871   0.       0.006   -0.006    1.5259  -0.5444   0.4655  -0.6978 147.9871]]
    15 1.5258789e-05 
    [[  0.1635  -0.0328  -0.0066   0.0006   0.0015   0.0004  -0.       0.       0.1635  -0.0328  -0.0066   0.0006]
     [  0.9032   0.383    0.1935   1.       0.006    0.003    0.0164   0.       0.9032   0.383    0.1935   1.    ]
     [ -0.4274   0.8432   0.326  174.5106  -0.006    0.       0.0209   1.5259  -0.4274   0.8432   0.326  174.5106]]





Adopting v3 reduces number of deviants from 54 to 39::


     13 static __device__ __inline__ float boundary_sample_reciprocal_domain(const float& u)
     14 {
     15     // return wavelength, from uniform sampling of 1/wavelength[::-1] domain
     16     // need to flip to match Geant4 energy sampling, see boundary_lookup.py 
     17     //float iw = lerp( boundary_domain_reciprocal.x , boundary_domain_reciprocal.y, u ) ;
     18     float iw = lerp( boundary_domain_reciprocal.y , boundary_domain_reciprocal.x, u ) ;
     19     return 1.f/iw ;
     20 }
     21 
     22 static __device__ __inline__ float boundary_sample_reciprocal_domain_v3(const float& u)
     23 {
     24     // see boundary_lookup.py
     25     float a = boundary_domain.x ;
     26     float b = boundary_domain.y ;
     27     return a*b/lerp( a, b, u ) ;
     28 }


::

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/ckm/ckm-so.py
     ckm-xx- comparing so.npy and ox.npy between two dirs 
    pwd /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1
       Sat Sep  8 21:11:11 CST 2018
    a  -rw-r--r--  1 blyth  staff  14224 Sep  8 21:11 source/evt/g4live/natural/-1/so.npy
    b  -rw-r--r--  1 blyth  staff  14224 Sep  8 21:11 source/evt/g4live/natural/1/ox.npy
    a (221, 4, 4) 
    b (221, 4, 4) 
    max deviation 6.1035156e-05 
     deviations above cut 1e-05 num_wh 39
    11 1.5258789e-05 
    [[  0.3261  -0.0653  -0.0131   0.0012   0.003    0.       0.       0.       0.3261  -0.0653  -0.0131   0.0012]
     [  0.8373   0.1958  -0.5104   1.       0.       0.006   -0.006    0.       0.8373   0.1958  -0.5104   1.    ]
     [ -0.5159   0.5917  -0.6194 173.2261  -0.006    0.0119  -0.006   -1.5259  -0.5159   0.5917  -0.6194 173.2261]]
    18 3.0517578e-05 
    [[  0.2963  -0.0594  -0.0119   0.0011   0.       0.0004   0.       0.       0.2963  -0.0594  -0.0119   0.0011]
     [  0.8952   0.4147  -0.1633   1.       0.       0.0089   0.       0.       0.8952   0.4147  -0.1633   1.    ]
     [ -0.4382   0.8857  -0.1532 277.77    -0.003    0.006    0.0015   3.0518  -0.4382   0.8857  -0.1532 277.7699]]
    22 1.5258789e-05 
    [[  0.2308  -0.0463  -0.0093   0.0009   0.       0.0004  -0.0001   0.       0.2308  -0.0463  -0.0093   0.0009]



