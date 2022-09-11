ct_scan_nmskTailInner
========================


Observe some rare spurious halo beyond the expected face of nmskTailInner.

::

    c
    ./ct.sh ana

    In [11]: w = s.simtrace[:,1,0] > 260.     

    In [15]: np.where(w)
    Out[15]: (array([216852, 349933, 387116, 615829]),)

    In [17]: s.simtrace[w,:3]
    Out[17]: 
    array([[[   0.   ,    0.   ,    1.   ,  395.232],
            [ 267.011,    0.   ,  -38.   ,    0.   ],
            [-128.   ,    0.   ,  -51.2  ,    1.   ]],

           [[   0.   ,    0.   ,    1.   ,  210.675],
            [ 261.461,    0.   ,  -38.   ,    0.   ],
            [  51.2  ,    0.   ,  -51.2  ,    1.   ]],

           [[   0.   ,    0.   ,    1.   ,  162.043],
            [ 263.904,    0.   ,  -38.   ,    0.   ],
            [ 102.4  ,    0.   ,  -51.2  ,    1.   ]],

           [[   0.   ,    0.   ,   -1.   ,  149.407],
            [ 260.668,    0.   ,  -39.3  ,    0.   ],
            [ 409.6  ,    0.   ,  -51.2  ,    1.   ]]], dtype=float32)


Problem intersect ray directions are close to, but not quite horizontal:: 

    In [19]: s.simtrace[w,3,:3]
    Out[19]: 
    array([[ 0.999,  0.   ,  0.033],
           [ 0.998,  0.   ,  0.063],
           [ 0.997,  0.   ,  0.081],
           [-0.997,  0.   ,  0.08 ]], dtype=float32)


Using simtrace selection to show the intersects leading to unexpected intersects.

CSG/tests/CSGSimtraceTest.py::

     58     if not s is None:
     59         sts = s.simtrace[s.simtrace[:,1,0] > 257.]
     60     else:
     61         sts = None
     62     pass
     63     if not sts is None:
     64         mpplt_simtrace_selection_line(ax, sts, axes=fr.axes, linewidths=2)
     65     pass


Seems to show the spurious are caused by missing intersects with the thin edge of 
the tubs nmskTailInnerITube.

AHHA, the translation uses disc when it should be using tubs::

    gc
    ./mtranslate.sh  

    2022-09-11 15:24:19.032 INFO  [3749623] [CSGGeometry::init_selection@174]  no SXYZ or SXYZW selection 
    2022-09-11 15:24:19.032 INFO  [3749623] [CSGDraw::draw@57] GeoChain::convertSolid converted CSGNode tree axis Z
    2022-09-11 15:24:19.032 INFO  [3749623] [CSGDraw::draw@58]  type 113 CSG::Name(type) disc IsTree 0 width 1 height 1

     di                           
    0                             
                                  
::

    022-09-11 15:24:19.027 INFO  [3749623] [X4SolidTree::Draw@61] ]
    2022-09-11 15:24:19.027 INFO  [3749623] [*X4PhysicalVolume::ConvertSolid_@1108] [ 0 soname nmskTail_inner_PartI_Tube lvname nmskTail_inner_PartI_Tube
    2022-09-11 15:24:19.027 INFO  [3749623] [X4Solid::Banner@86]  lvIdx     0 soIdx     0 soname nmskTail_inner_PartI_Tube lvname nmskTail_inner_PartI_Tube
    2022-09-11 15:24:19.027 INFO  [3749623] [*X4Solid::Convert@109] [ convert nmskTail_inner_PartI_Tube lvIdx 0
    2022-09-11 15:24:19.027 INFO  [3749623] [X4Solid::init@185] [ X4SolidBase identifier a entityType                   25 entityName               G4Tubs name                nmskTail_inner_PartI_Tube root 0x0
    2022-09-11 15:24:19.027 INFO  [3749623] [X4Solid::convertTubs@1050]  has_deltaPhi 0 pick_disc 1 deltaPhi_segment_enabled 1 is_x4tubsnudgeskip 0 do_nudge_inner 1
    2022-09-11 15:24:19.027 INFO  [3749623] [X4Solid::init@221] ]
    2022-09-11 15:24:19.027 INFO  [3749623] [*X4Solid::Convert@127]  hint_external_bbox  0 expect_external_bbox 0 set_external_bbox  0
    2022-09-11 15:24:19.027 INFO  [3749623] [*X4Solid::Convert@138] ]
    2022-09-11 15:24:19.028 INFO  [3749623] [NTreeProcess<nnode>::init@159]  NOT WITH_CHOPPER 
    2022-09-11 15:24:19.028 INFO  [3749623] [NTreeProcess<nnode>::init@165]  want_to_balance NO y when height0 exceeds MaxHeight0  balancer.height0 0 MaxHeight0 3
    2022-09-11 15:24:19.028 INFO  [3749623] [*X4PhysicalVolume::ConvertSolid_FromRawNode@1156]  after NTreeProcess:::Process 
    2022-09-11 15:24:19.028 INFO  [3749623] [*X4PhysicalVolume::ConvertSolid_FromRawNode@1165] [ before NCSG::Adopt 
    2022-09-11 15:24:19.028 INFO  [3749623] [*NCSG::Adopt@165]  [  soIdx 0 lvIdx 0
    2022-09-11 15:24:19.028 INFO  [3749623] [*NCSG::MakeNudger@276]  treeidx 0 nudgeskip 0




* nmskTailOuterITube zrange 0.15 -0.15  : 0.30
* nmskTailOuter lip zrange -39.00 -39.30

* nmskTailInnerITube  0.65 -0.65  : 1.30
* nmskTailInner lip zrange  -38.00 -39.30

* both the lips have hz less than 1mm so they are getting translated as disc 
* THIS EXPLAINS THE LACK OF EDGE INTERSECTS 


::

    0986 const float X4Solid::hz_disc_cylinder_cut = 1.f ; // 1mm 


    1022 void X4Solid::convertTubs()
    1023 { 
    1024     const G4Tubs* const solid = static_cast<const G4Tubs*>(m_solid);
    1025     assert(solid);
    1026     //LOG(info) << "\n" << *solid ; 
    1027 
    1028     // better to stay double until there is a need to narrow to float for storage or GPU 
    1029     double hz = solid->GetZHalfLength()/mm ;
    1030     double  z = hz*2.0 ;   // <-- this full-length z is what GDML stores
    1031 
    1032     double startPhi = solid->GetStartPhiAngle()/degree ;
    1033     double deltaPhi = solid->GetDeltaPhiAngle()/degree ;
    1034     double rmax = solid->GetOuterRadius()/mm ;
    1035 
    1036     bool pick_disc = hz < hz_disc_cylinder_cut ;
    1037 
    1038     bool is_x4tubsnudgeskip = isX4TubsNudgeSkip()  ;
    1039     bool do_nudge_inner = is_x4tubsnudgeskip ? false : true ;   // --x4tubsnudgeskip 0,1,2  # lvIdx of the tree 
    1040 
    1041     nnode* tube = pick_disc ? convertTubs_disc() : convertTubs_cylinder(do_nudge_inner) ;
    1042 
    1043     bool deltaPhi_segment_enabled = true ;
    1044     bool has_deltaPhi = deltaPhi < 360. ;
    1045 



