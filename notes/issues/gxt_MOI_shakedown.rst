gxt_MOI_shakedown
===================


Next
------

* :doc:`NNVTMCPPMTsMask_virtual_G4Polycone_hatbox_spurious_intersects`


Setup
-------

::

    epsilon:g4cx blyth$ ./gxt.sh grab
    epsilon:g4cx blyth$ ./gxs.sh grab
    epsilon:g4cx blyth$ ./gxt.sh ana
    epsilon:g4cx blyth$ MASK=t ./gxt.sh ana




FIXED Issue 2 : gxt overlay of gxs input photon intersects appear mid CD
----------------------------------------------------------------------------

* problem was due to update to J003 not being accomodated by OPTICKS_INPUT_PHOTON_FRAME setting in "com_"


::

     60     t = Fold.Load(symbol="t")
     61     a = Fold.Load("$A_FOLD", symbol="a")
     62     b = Fold.Load("$B_FOLD", symbol="b")
     63     print("cf.cfbase : %s " % cf.cfbase)
     64 
     65     print("---------Fold.Load.done")
     66     x = a
     67 
     68     print(repr(t))
     69     print(repr(a))
     70     print(repr(b))
     71 
     72     print("---------print.done")
     73 
     74 
     75     if not a is None and not a.seq is None:
     76         a_nib = seqnib_(a.seq[:,0])                  # valid steppoint records from seqhis count_nibbles
     77         a_gpos_ = a.record[PIDX,:a_nib[PIDX],0,:3]  # global frame photon step record positions of single PIDX photon
     78         a_gpos  = np.ones( (len(a_gpos_), 4 ) )
     79         a_gpos[:,:3] = a_gpos_
     80         a_lpos = np.dot( a_gpos, t.sframe.w2m )
     81     else:
     82         a_lpos = None
     83     pass
     84 
     ..
     95     x_lpos = a_lpos



::

    In [1]: print(os.environ["A_FOLD"])
    /Users/blyth/.opticks/ntds3/G4CXOpticks/G4CXSimulateTest/ALL

    In [2]: print(os.environ["B_FOLD"])
    /Users/blyth/.opticks/ntds3/G4CXOpticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL


    In [7]: seqhis_(a.seq[:,0]) 
    Out[7]: 
    ['TO BT BT SA',
     'TO BT BT SA',
     'TO AB',
     'TO SC BT BT BT BT BT BT SD',
     'TO AB',
     'TO AB',
     'TO BT BT SA',
     'TO SC BT BT BT BR BT BT BT BT',
     'TO BT BT DR BT BT AB',
     'TO SC AB',

    In [9]: seqnib_(a.seq[:10,0])
    Out[9]: array([ 4,  4,  2,  9,  2,  2,  4, 10,  7,  3], dtype=uint64)


Hmm this looks like the input photon frame is defaulting to global frame::

    In [12]: a_gpos
    Out[12]: 
    array([[    19.525,      0.   ,    999.   ,      1.   ],
           [    19.525,      0.   , -17699.99 ,      1.   ],
           [    19.526,      0.   , -17823.988,      1.   ],
           [    19.318,      0.   , -19628.99 ,      1.   ]])


Then transforming into the local gxt frame results in coordinates nowhere near it::

    In [14]: np.dot( a_gpos, t.sframe.w2m )
    Out[14]: 
    array([[   565.931,    -18.684,  18610.745,      1.   ],
           [ -9939.331,    -18.684,  34079.801,      1.   ],
           [-10008.994,    -18.685,  34182.38 ,      1.   ],
           [-11023.11 ,    -18.486,  35675.565,      1.   ]])

    In [15]: a_lpos
    Out[15]: 
    array([[   565.931,    -18.684,  18610.745,      1.   ],
           [ -9939.331,    -18.684,  34079.801,      1.   ],
           [-10008.994,    -18.685,  34182.38 ,      1.   ],
           [-11023.11 ,    -18.486,  35675.565,      1.   ]])



::

     75     if not a is None and not a.seq is None:
     76         a_nib = seqnib_(a.seq[:,0])                  # valid steppoint records from seqhis count_nibbles
     77         a_gpos_ = a.record[PIDX,:a_nib[PIDX],0,:3]   # global frame photon step record positions of single PIDX photon
     78         a_gpos  = np.ones( (len(a_gpos_), 4 ) )
     79         a_gpos[:,:3] = a_gpos_
     80         a_lpos = np.dot( a_gpos, t.sframe.w2m )      # a global positions into gxt target frame 
     81     else:
     82         a_lpos = None
     83     pass



gxs.sh OPTICKS_INPUT_PHOTON_FRAME ?
----------------------------------------

HMM, OPTICKS_INPUT_PHOTON_FRAME blank first and then gets set to NNVT:0:1000 by COMMON.sh::

    epsilon:g4cx blyth$ ./gxs.sh info
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                       TMP_GEOMDIR : /tmp/blyth/opticks/J003 
                           GEOMDIR : /Users/blyth/.opticks/ntds3/G4CXOpticks 

                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON_.sh
                              GEOM : J003
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy
      OPTICKS_INPUT_PHOTON_ABSPATH : /Users/blyth/.opticks/InputPhotons/DownXZ1000_f8.npy
        OPTICKS_INPUT_PHOTON_LABEL : DownXZ1000
                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON.sh 
                         ScriptDir : ./../bin 
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy 
        OPTICKS_INPUT_PHOTON_FRAME :  
      OPTICKS_INPUT_PHOTON_ABSPATH : /Users/blyth/.opticks/InputPhotons/DownXZ1000_f8.npy 

                       BASH_SOURCE : ./../bin/COMMON.sh
                              GEOM : J003
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy
        OPTICKS_INPUT_PHOTON_FRAME : NNVT:0:1000
                               MOI : NNVT:0:1000
             BASH_SOURCE : ./gxs.sh 
                  gxsdir : . 
                    GEOM : J003 
                 GEOMDIR : /Users/blyth/.opticks/ntds3/G4CXOpticks 
                  CFBASE :  
                    BASE : /Users/blyth/.opticks/ntds3/G4CXOpticks/G4CXSimulateTest 
                   UBASE : .opticks/ntds3/G4CXOpticks/G4CXSimulateTest 
                    FOLD : /Users/blyth/.opticks/ntds3/G4CXOpticks/G4CXSimulateTest/ALL 
    OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy 
    epsilon:g4cx blyth$ 


* HMM: the value in use should be held in metadata ?

::

    epsilon:issues blyth$ opticks-f OPTICKS_INPUT_PHOTON_FRAME
    ./CSG/tests/CSGFoundry_getFrame_Test.sh:export OPTICKS_INPUT_PHOTON_FRAME="Hama:0:1000"
    ./bin/COMMON.sh:     J000) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
    ./bin/COMMON.sh:     J001) OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000 ;;
    ./bin/COMMON.sh:     J002) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
    ./bin/COMMON.sh:     J003) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
    ./bin/COMMON.sh:   [ -n "$OPTICKS_INPUT_PHOTON_FRAME" ] && export OPTICKS_INPUT_PHOTON_FRAME
    ./bin/COMMON.sh:   [ -n "$OPTICKS_INPUT_PHOTON_FRAME" ] && export MOI=$OPTICKS_INPUT_PHOTON_FRAME
    ./bin/COMMON.sh:    vars="BASH_SOURCE GEOM OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME MOI"
    ./bin/OPTICKS_INPUT_PHOTON.sh:OPTICKS_INPUT_PHOTON_FRAME
    ./bin/OPTICKS_INPUT_PHOTON.sh:    vars="BASH_SOURCE ScriptDir OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME OPTICKS_INPUT_PHOTON_ABSPATH"
    ./sysrap/SEventConfig.hh:    static constexpr const char* kInputPhotonFrame = "OPTICKS_INPUT_PHOTON_FRAME" ; 
    ./sysrap/tests/SEvtTest.sh:export OPTICKS_INPUT_PHOTON_FRAME=0 
    ./u4/tests/U4RecorderTest.cc:    // The frame is needed for transforming input photons when using OPTICKS_INPUT_PHOTON_FRAME. 
    epsilon:opticks blyth$ 

::

    const char* SEventConfig::InputPhotonFrame(){   return _InputPhotonFrame ; }


    epsilon:sysrap blyth$ opticks-f SEventConfig::InputPhotonFrame
    ./CSG/tests/CSGFoundry_getFrame_Test.cc:    const char* ipf_ = SEventConfig::InputPhotonFrame(); 
    ./bin/OPTICKS_INPUT_PHOTON.sh:   moi_or_iidx string eg "Hama:0:1000" OR "35000", default of SEventConfig::InputPhotonFrame
    ./sysrap/SCF.h:    const char* ipf_ = SEventConfig::InputPhotonFrame(); 
    ./sysrap/SEventConfig.cc:const char* SEventConfig::InputPhotonFrame(){   return _InputPhotonFrame ; }
    ./sysrap/tests/SEvtTest.cc:    const char* ipf = SEventConfig::InputPhotonFrame();  
    ./g4cx/G4CXOpticks.cc:        const char* ipf = SEventConfig::InputPhotonFrame();
    epsilon:opticks blyth$ 


    300 void G4CXOpticks::simulate()
    301 {
    302 #ifdef __APPLE__
    303      LOG(fatal) << " APPLE skip " ;
    304      return ;
    305 #endif
    306     LOG(LEVEL) << "[" ;
    307     LOG(LEVEL) << desc() ;
    308     assert(cx);
    309     assert(qs);
    310     assert( SEventConfig::IsRGModeSimulate() );
    311 
    312 
    313     SEvt* sev = SEvt::Get();  assert(sev);
    314 
    315     bool has_input_photon = sev->hasInputPhoton() ;
    316     if(has_input_photon)
    317     {
    318         const char* ipf = SEventConfig::InputPhotonFrame();
    319         sframe fr = fd->getFrame(ipf) ;
    320         sev->setFrame(fr);
    321     }
    322 
    323     unsigned num_genstep = sev->getNumGenstepFromGenstep();
    324     unsigned num_photon  = sev->getNumPhotonFromGenstep();
    325 


    2815 const char* CSGFoundry::FRS = "-1" ;
    2816 
    2817 sframe CSGFoundry::getFrame() const
    2818 {   
    2819     const char* moi_or_iidx = SSys::getenvvar("MOI",FRS);   // TODO: MOI->FRS perhaps ?
    2820     return getFrame(moi_or_iidx);
    2821 }
    2822 sframe CSGFoundry::getFrame(const char* frs) const
    2823 {   
    2824     sframe fr ; 
    2825     int rc = getFrame(fr, frs ? frs : FRS ); 
    2826     if(rc != 0) LOG(error) << " frs " << frs << std::endl << getFrame_NOTES ;
    2827     if(rc != 0) std::raise(SIGINT);
    2828 
    2829     fr.prepare();  // creates Tran<double>
    2830     return fr ;
    2831 }

    2862 int CSGFoundry::getFrame(sframe& fr, const char* frs ) const
    2863 {
    2864     int rc = 0 ;
    2865     bool looks_like_moi = SStr::StartsWithLetterAZaz(frs) || strstr(frs, ":") || strcmp(frs,"-1") == 0 ;
    2866     if(looks_like_moi)
    2867     {
    2868         int midx, mord, iidx ;  // mesh-index, mesh-ordinal, gas-instance-index
    2869         parseMOI(midx, mord, iidx,  frs );
    2870         rc = getFrame(fr, midx, mord, iidx);
    2871     }
    2872     else
    2873     {
    2874          int inst_idx = SName::ParseIntString(frs, 0) ;
    2875          rc = getFrame(fr, inst_idx);
    2876     }
    2877 
    2878     fr.set_propagate_epsilon( SEventConfig::PropagateEpsilon() );
    2879     fr.frs = strdup(frs);
    2880     LOG(LEVEL) << " fr " << fr ;    // no grid has been set at this stage, just ce,m2w,w2m
    2881     if(rc != 0) LOG(error) << "Failed to lookup frame with frs [" << frs << "] looks_like_moi " << looks_like_moi  ;
    2882     return rc ;
    2883 }




    In [2]: a.sframe 
    Out[2]: 
    sframe       : 
    path         : /Users/blyth/.opticks/ntds3/G4CXOpticks/G4CXSimulateTest/ALL/sframe.npy
    meta         : {'creator': 'sframe::save', 'frs': '-1'}
    ce           : array([    0.,     0.,     0., 60000.], dtype=float32)
    grid         : ix0    0 ix1    0 iy0    0 iy1    0 iz0    0 iz1    0 num_photon    0 gridscale     0.0000
    bbox         : array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    target       : midx      0 mord      0 iidx      0       inst       0   
    qat4id       : ins_idx     -1 gas_idx   -1   -1 
    m2w          : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    w2m          : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    id           : 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)
    ins_gas_ias  :  ins      0 gas    0 ias    0 


::

    In [5]: a.sframe.meta.frs
    Out[5]: '-1'


After gxs rerun and grab, the gxs record points are landing on the gxt targetted PMT::

    gx
    ./gxs.sh        # workstation
    ./gxs.sh grab   # laptop
    ./gxt.sh ana    # laptop


    In [1]: a.sframe.meta.frs
    Out[1]: 'NNVT:0:1000'



FIXED Issue 1 :  No longer need MASK=t OR MASK=non to make the simtrace intersects visible 
---------------------------------------------------------------------------------------------

::

    epsilon:g4cx blyth$ ./gxt.sh grab
    epsilon:g4cx blyth$ ./gxt.sh ana
    epsilon:g4cx blyth$ MASK=t ./gxt.sh ana


./gxt.sh ana
~~~~~~~~~~~~~~

* pv plot starts all black, zooming out see only the cegs grid rectangle of gs positions 
* mp plot stars all white, no easy way to zoom out  

MASK=t ./gxt.sh ana
~~~~~~~~~~~~~~~~~~~~~~

* pv plot immediately shows the simtrace isect of the ~7 PMTs 
* zooming out see lots more 
* also zooming out more see the genstep grid rectangle, 
  which is greatly offset from the intersects

* mp plot, blank white again but lots of key entries


gx/tests/G4CXSimtraceTest.py 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The genstep transform looks to be carrying the 4th column identity info::

    In [3]: t.genstep[0]
    Out[3]: 
    array([[    0.   ,     0.   ,       nan,     0.   ],
           [    0.   ,     0.   ,     0.   ,     1.   ],
           [    0.24 ,    -0.792,     0.562,     0.   ],
           [   -0.957,    -0.29 ,     0.   ,     0.   ],
           [    0.163,    -0.538,    -0.827,     0.   ],
           [-3354.313, 11057.688, 16023.353,    -0.   ]], dtype=float32)

        
Add the gs_tran 4th column fixup in ana/framegensteps.py::

     64         ## apply the 4x4 transform in rows 2: to the position in row 1 
     65         world_frame_centers = np.zeros( (len(gs), 4 ), dtype=np.float32 )
     66         for igs in range(len(gs)): 
     67             gs_pos = gs[igs,1]          ## normally origin (0,0,0,1)
     68             gs_tran = gs[igs,2:]        ## m2w with grid translation 
     69             gs_tran[:,3] = [0,0,0,1]   ## fixup 4th column, as may contain identity info
     70             world_frame_centers[igs] = np.dot( gs_pos, gs_tran )    
     71             #   world_frame_centers = m2w * grid_translation * model_frame_positon
     72         pass


* the "fixup 4th column" gets the genstep grid to correspond to the intersects and no longer need MASK=t 
  to see intersects 




