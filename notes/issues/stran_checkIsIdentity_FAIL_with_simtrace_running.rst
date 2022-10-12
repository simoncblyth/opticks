stran_checkIsIdentity_FAIL_with_simtrace_running
=======================================================

* from :doc:`simtrace_over_1M_unchecked_against_size_of_CurandState`


The problem with the transforms pairing appears to be explained by a lack of
a 1. in position (3,3).   

* TODO: investigate the sframe.h source of the transform, is the lack of the 1. expected ? 


Capture backtrace from checkIsIdenity issue::

    N[blyth@localhost opticks]$ env | grep SIGINT
    stran_checkIsIdentity_SIGINT=1

    N[blyth@localhost opticks]$ MOI=Hama:0:1000 ~/opticks/g4cx/gxt.sh dbg 
    === cehigh : GEOM J004 MOI Hama:0:1000
    === cehigh_PMT
    CEHIGH_0=-8:8:0:0:-6:-4:1000:4
    gdb -ex r --args G4CXSimtraceTest -ex r
    Wed Oct 12 19:01:37 CST 2022

    stran.h : Tran::checkIsIdentity FAIL :  caller FromPair epsilon 1e-06 mxdif_from_identity 12075.9

    Program received signal SIGINT, Interrupt.
    0x00007fffecd0b4fb in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007fffecd0b4fb in raise () from /lib64/libpthread.so.0
    #1  0x00007fffed64c502 in Tran<double>::checkIsIdentity (this=0x4027090, mat=105 'i', caller=0x7fffed72397c "FromPair", epsilon=9.9999999999999995e-07)
        at /data/blyth/junotop/opticks/sysrap/stran.h:638
    #2  0x00007fffed64b7f0 in Tran<double>::FromPair (t=0x2dd6cc0, v=0x2dd6d00, epsilon=9.9999999999999995e-07) at /data/blyth/junotop/opticks/sysrap/stran.h:712
    #3  0x00007fffed65ef4a in SFrameGenstep::MakeCenterExtentGensteps (fr=...) at /data/blyth/junotop/opticks/sysrap/SFrameGenstep.cc:160
    #4  0x00007fffed676682 in SEvt::setFrame (this=0x2dd6bf0, fr=...) at /data/blyth/junotop/opticks/sysrap/SEvt.cc:269
    #5  0x00007ffff7b8fcfb in G4CXOpticks::simtrace (this=0x7fffffff57a0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:391
    #6  0x0000000000408d52 in main (argc=3, argv=0x7fffffff5908) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimtraceTest.cc:27
    (gdb) 


::

    377 void G4CXOpticks::simtrace()
    378 {
    388     SEvt* sev = SEvt::Get();  assert(sev);
    390     sframe fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 
    391     sev->setFrame(fr);
    393     cx->setFrame(fr);
    397     qs->simtrace();
    399 }

    0256 void SEvt::setFrame(const sframe& fr )
     257 {
     258     frame = fr ;
     259 
     260     if(SEventConfig::IsRGModeSimtrace())
     261     {
     262         const char* frs = fr.get_frs() ; // nullptr when default -1 : meaning all geometry 
     263         if(frs)
     264         {
     265             LOG(LEVEL) << " non-default frs " << frs << " passed to SEvt::setReldir " ;
     266             setReldir(frs);
     267         }
     268 
     269         NP* gs = SFrameGenstep::MakeCenterExtentGensteps(frame);
     270         LOG(LEVEL) << " simtrace gs " << ( gs ? gs->sstr() : "-" ) ;
     271         addGenstep(gs);
     272 
     273         if(frame.is_hostside_simtrace()) setFrame_HostsideSimtrace();
     274 


    137 NP* SFrameGenstep::MakeCenterExtentGensteps(sframe& fr)
    138 {
    139     const float4& ce = fr.ce ;
    140     float gridscale = SSys::getenvfloat("GRIDSCALE", 0.1 ) ;
    141 
    142     // CSGGenstep::init
    143     std::vector<int> cegs ;
    144     GetGridConfig(cegs, "CEGS", ':', "16:0:9:1000" );
    145     fr.set_grid(cegs, gridscale);
    146 
    147     std::vector<float3> ce_offset ;
    148     CE_OFFSET(ce_offset, ce);
    149 
    150     LOG(LEVEL)
    151         << " ce " << ce
    152         << " ce_offset.size " << ce_offset.size()
    153         ;
    154 
    155 
    156     int ce_scale = SSys::getenvint("CE_SCALE", 1) ; // TODO: ELIMINATE AFTER RTP CHECK 
    157     LOG_IF(fatal, ce_scale == 0) << "warning CE_SCALE is not enabled : NOW THINK THIS SHOULD ALWAYS BE ENABLED " ;
    158 
    159 
    160     Tran<double>* geotran = Tran<double>::FromPair( &fr.m2w, &fr.w2m, 1e-6 );
    161 
    162 
    163     std::vector<NP*> gsl ;
    164     NP* gs_base = MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );
    165     gsl.push_back(gs_base) ;
    166 


Save the unexpected transforms:: 

    +    if(!ok) 
    +    {
    +         std::cerr << "stran.h Tran::FromPair checkIsIdentity FAIL " << std::endl ; 
    +         const char* path = "/tmp/stran_FromPair_checkIsIdentity_FAIL.npy" ; 
    +         std::cerr << "stran.h save to path " << path << std::endl ; 
    +         tr->save_(path); 
    +    }




    In [1]: a = np.load("/tmp/stran_FromPair_checkIsIdentity_FAIL.npy")

    In [2]: a                                                         
    Out[2]: 
    array([[[     0.48 ,     -0.379,      0.792,      0.   ],
            [    -0.619,     -0.785,      0.   ,      0.   ],
            [     0.621,     -0.49 ,     -0.611,      0.   ],
            [-12075.873,   9528.691,  11876.771,      0.   ]],

           [[     0.48 ,     -0.619,      0.621,      0.   ],
            [    -0.379,     -0.785,     -0.49 ,      0.   ],
            [     0.792,      0.   ,     -0.611,      0.   ],
            [    -0.006,     -0.009,  19434.   ,      0.   ]],

           [[     1.   ,      0.   ,     -0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ,     -0.   ],
            [    -0.   ,      0.   ,      1.   ,      0.   ],
            [ 12075.872,  -9528.691, -11876.771,      0.   ]]])


    2022-10-13 03:07:54.595 INFO  [37141] [CSGTarget::getGlobalCenterExtent@260] 
    t:[   0.480   -0.379    0.792    0.000 ][  -0.619   -0.785    0.000    0.000 ][   0.621   -0.490   -0.611    0.000 ][-12075.873 9528.691 11876.771    0.000 ]
    v:[   0.480   -0.619    0.621    0.000 ][  -0.379   -0.785   -0.490    0.000 ][   0.792    0.000   -0.611    0.000 ][  -0.006   -0.009 19434.000    0.000 ]
    2022-10-13 03:07:54.595 INFO  [37141] [CSGTarget::getGlobalCenterExtent@286]  
    q ( 0.480,-0.379, 0.792, 0.000) (-0.619,-0.785, 0.000, 0.000) ( 0.621,-0.490,-0.611, 0.000) (-12075.873,9528.691,11876.771, 1.000)  
    ins_idx 39216 gas_idx 3 sensor_identifier 3354 sensor_index 3354






    In [3]: a[:,3,3]
    Out[3]: array([0., 0., 0.])

    In [4]: a[:,3,3] = 1

    In [5]: a
    Out[5]: 
    array([[[     0.48 ,     -0.379,      0.792,      0.   ],
            [    -0.619,     -0.785,      0.   ,      0.   ],
            [     0.621,     -0.49 ,     -0.611,      0.   ],
            [-12075.873,   9528.691,  11876.771,      1.   ]],

           [[     0.48 ,     -0.619,      0.621,      0.   ],
            [    -0.379,     -0.785,     -0.49 ,      0.   ],
            [     0.792,      0.   ,     -0.611,      0.   ],
            [    -0.006,     -0.009,  19434.   ,      1.   ]],

           [[     1.   ,      0.   ,     -0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ,     -0.   ],
            [    -0.   ,      0.   ,      1.   ,      0.   ],
            [ 12075.872,  -9528.691, -11876.771,      1.   ]]])


    In [6]: np.dot( a[0], a[1] )
    Out[6]: 
    array([[ 1.   , -0.   ,  0.   ,  0.   ],
           [-0.   ,  1.   , -0.   , -0.   ],
           [ 0.   , -0.   ,  1.   ,  0.   ],
           [-0.   ,  0.   , -0.001,  1.   ]])

    In [7]: np.dot( a[1], a[0] )
    Out[7]: 
    array([[ 1.   ,  0.   , -0.   ,  0.   ],
           [ 0.   ,  1.   ,  0.   , -0.   ],
           [-0.   ,  0.   ,  1.   ,  0.   ],
           [-0.001,  0.   ,  0.001,  1.   ]])



    In [8]: a[:,3,3] = 0        

    In [9]: np.dot( a[0], a[1] )
    Out[9]: 
    array([[     1.   ,     -0.   ,      0.   ,      0.   ],
           [    -0.   ,      1.   ,     -0.   ,     -0.   ],
           [     0.   ,     -0.   ,      1.   ,      0.   ],
           [     0.005,      0.009, -19434.001,     -0.   ]])

    In [10]: np.dot( a[1], a[0] )
    Out[10]: 
    array([[     1.   ,      0.   ,     -0.   ,      0.   ],
           [     0.   ,      1.   ,      0.   ,     -0.   ],
           [    -0.   ,      0.   ,      1.   ,      0.   ],
           [ 12075.872,  -9528.691, -11876.771,      0.   ]])



::

    2919 int CSGFoundry::getFrame(sframe& fr, const char* frs ) const
    2920 {
    2921     int rc = 0 ;
    2922     bool looks_like_moi = SStr::StartsWithLetterAZaz(frs) || strstr(frs, ":") || strcmp(frs,"-1") == 0 ;
    2923     if(looks_like_moi)
    2924     {
    2925         int midx, mord, iidx ;  // mesh-index, mesh-ordinal, gas-instance-index
    2926         parseMOI(midx, mord, iidx,  frs );
    2927         rc = getFrame(fr, midx, mord, iidx);
    2928     }
    2929     else
    2930     {
    2931          int inst_idx = SName::ParseIntString(frs, 0) ;
    2932          rc = getFrame(fr, inst_idx);
    2933     }
    2934 
    2935     fr.set_propagate_epsilon( SEventConfig::PropagateEpsilon() );
    2936     fr.frs = strdup(frs);
    2937     LOG(LEVEL) << " fr " << fr ;    // no grid has been set at this stage, just ce,m2w,w2m
    2938     LOG_IF(error, rc != 0) << "Failed to lookup frame with frs [" << frs << "] looks_like_moi " << looks_like_moi  ;
    2939     return rc ;
    2940 }


    2942 int CSGFoundry::getFrame(sframe& fr, int inst_idx) const
    2943 {
    2944     return target->getFrame( fr, inst_idx );
    2945 }
    2946 
    2947 
    2948 
    2949 int CSGFoundry::getFrame(sframe& fr, int midx, int mord, int iidxg) const
    2950 {
    2951     int rc = 0 ;
    2952     if( midx == -1 )
    2953     {
    2954         unsigned long long emm = 0ull ;   // hmm instance var ?
    2955         iasCE(fr.ce, emm);
    2956     }
    2957     else
    2958     {
    2959         rc = target->getFrame( fr, midx, mord, iidxg );
    2960     }
    2961     return rc ;
    2962 }

    115 int CSGTarget::getFrame(sframe& fr,  int midx, int mord, int iidxg ) const
    116 {
    117     fr.set_midx_mord_iidx( midx, mord, iidxg );
    118     return getCenterExtent( fr.ce, midx, mord, iidxg, &fr.m2w , &fr.w2m );
    119 }

    138 int CSGTarget::getFrame(sframe& fr, int inst_idx ) const
    139 {
    140     const qat4* _t = foundry->getInst(inst_idx);
    141 
    142     int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
    143     _t->getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );
    144 
    145     assert( ins_idx == inst_idx );
    146     fr.set_inst(inst_idx);
    147   
    148     // HMM: these values are already there inside the matrices ? 
    149     fr.set_identity(ins_idx, gas_idx, sensor_identifier, sensor_index ) ;
    150 
    151     qat4 t(_t->cdata());   // copy the instance (transform and identity info)
    152     const qat4* v = Tran<double>::Invert(&t);     // identity gets cleared in here 
    153 
    154     qat4::copy(fr.m2w,  t);
    155     qat4::copy(fr.w2m, *v);
    156 
    157     const CSGSolid* solid = foundry->getSolid(gas_idx);
    158     fr.ce = solid->center_extent ;
    159 
    160     return 0 ;
    161 }



::

    In [5]: (np.dot( a[0], a[1] ) - np.eye(4)).max()
    Out[5]: 0.00013393204426392913



