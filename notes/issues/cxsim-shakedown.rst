cxsim-shakedown
==================

Standard geometry, simple carrier gensteps.

Remote::

    cx
    ./cxsim.sh 

Local::

    cx
    ./cxsim.sh grab
    ./cxsim.sh ana




water-water micro steps ?
----------------------------



Looks like repeated step records 4,5,6,7,8,9::

    r[0,:,:3]
    [[[     0.         0.         0.         0.   ]
      [     1.         0.         0.         1.   ]
      [     0.         1.         0.       500.   ]]

     [[ 13355.625      0.         0.        68.512]
      [    -0.517      0.371     -0.771      1.   ]
      [    -0.207     -0.929     -0.308    500.   ]]

     [[  2799.331   7577.046 -15749.353    173.253]
      [    -0.513      0.372     -0.773      1.   ]
      [     0.207     -0.821     -0.532    500.   ]]

     [[  2716.439   7637.113 -15874.207    174.078]
      [    -0.593      0.349     -0.726      1.   ]
      [     0.195     -0.812     -0.55     500.   ]]

     [[  1517.433   8343.771 -17343.04     183.436]
      [    -0.593      0.349     -0.726      1.   ]
      [     0.195     -0.812     -0.55     500.   ]]

     [[  1517.433   8343.771 -17343.04     183.436]
      [    -0.593      0.349     -0.726      1.   ]
      [     0.195     -0.812     -0.55     500.   ]]

     [[  1517.433   8343.771 -17343.04     183.436]
      [    -0.593      0.349     -0.726      1.   ]
      [     0.195     -0.812     -0.55     500.   ]]

     [[  1517.432   8343.771 -17343.04     183.436]
      [    -0.593      0.349     -0.726      1.   ]
      [     0.195     -0.812     -0.55     500.   ]]

     [[  1517.432   8343.771 -17343.04     183.436]
      [    -0.593      0.349     -0.726      1.   ]
      [     0.195     -0.812     -0.55     500.   ]]

     [[  1517.432   8343.771 -17343.04     183.436]
      [    -0.593      0.349     -0.726      1.   ]
      [     0.195     -0.812     -0.55     500.   ]]]


But the digests shows the steps are not the same::

    bflagdesc_(r[0,j])
          0 prd(  0    0     0 0)   TO               TO  :                      Galactic///Galactic : 71df83326df7316d984daac05b8ffe0d 
          0 prd( 20 2328     0 0)   SC            TO|SC  :                             Acrylic///LS : 844ee844f834dbea725b61b78d93c2c1 
          0 prd( 20 2328     0 0)   BT         TO|BT|SC  :                             Acrylic///LS : 0df28c0c3beb68b8bef3cb67dedcc8d8 
          0 prd( 19 2327     0 0)   BT         TO|BT|SC  :                          Water///Acrylic : 08d30e1e02c9861485618fc27c5010e1 
          0 prd( 27 3094 37684 1)   BT         TO|BT|SC  :                            Water///Water : 5d75ecd60a3c29e7ff8bb193772607f2 
          0 prd( 27 3094 37684 1)   BT         TO|BT|SC  :                            Water///Water : 87ad3cbbda5762b12d3acd3127a24c9c 
          0 prd( 27 3094 37684 1)   BT         TO|BT|SC  :                            Water///Water : 6d6b7d4098dbc89c951c9a5869f04fe5 
          0 prd( 27 3094 37684 1)   BT         TO|BT|SC  :                            Water///Water : 496ea46ace60ccfda0ffe3e249a87bbd 
          0 prd( 27 3094 37684 1)   BT         TO|BT|SC  :                            Water///Water : d83560cdb28f61855a156e053a37ea7e 
          0 prd( 27 3094 37684 1)   BT         TO|BT|SC  :                            Water///Water : 25ec76ba20801b63a07da733e26d1b7f 

Where does this identity info come form ?

ana/p.py::

     22 identity_ = lambda p:p.view(np.uint32)[3,1]
     23 primIdx_   = lambda p:identity_(p) >> 16
     24 instanceId_  = lambda p:identity_(p) & 0xffff
     25 
     26 idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff      ## HMM: looks to be always zero 
     27 orient_   = lambda p:p.view(np.uint32)[3,2] >> 31
     28 
     29 flagmask_ = lambda p:p.view(np.uint32)[3,3]
     30 
     31 flagdesc_ = lambda p:" %6d prd(%3d %4d %5d %1d)  %3s  %15s " % ( idx_(p),  boundary_(p),primIdx_(p),instanceId_(p), orient_(p),  hm.label(flag_(p)),hm.label( flagmask_(p) ))


::

    402 extern "C" __global__ void __closesthit__ch()
    403 {
    404     //unsigned instance_index = optixGetInstanceIndex() ;  0-based index within IAS
    405     unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build and InstanceId.h 
    406     unsigned prim_idx = optixGetPrimitiveIndex() ;  // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    407     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ;
    408 
    409 #ifdef WITH_PRD
    410     quad2* prd = getPRD<quad2>();
    411 
    412     prd->set_identity( identity ) ;
    413     //printf("//__closesthit__ch prd.boundary %d \n", prd->boundary() );  // boundary set in IS for WITH_PRD
    414     float3* normal = prd->normal();


    1145 inline QSIM_METHOD int qsim::propagate(const int bounce, sphoton& p, qstate& s, const quad2* prd, curandStateXORWOW& rng, unsigned idx )
    1146 {
    1147     const unsigned boundary = prd->boundary() ;
    1148     const unsigned identity = prd->identity() ;
    1149     const float3* normal = prd->normal();
    1150     float cosTheta = dot(p.mom, *normal ) ;    
    1151    
    1152 #ifdef DEBUG_COSTHETA
    1153     if( idx == pidx ) printf("//qsim.propagate idx %d bnc %d cosTheta %10.4f dir (%10.4f %10.4f %10.4f) nrm (%10.4f %10.4f %10.4f) \n",
    1154                  idx, bounce, cosTheta, p.mom.x, p.mom.y, p.mom.z, normal->x, normal->y, normal->z );
    1155 #endif
    1156 
    1157     p.set_prd(boundary, identity, cosTheta);
    1158 

    050 struct sphoton
     51 {
     52     float3 pos ;
     53     float  time ;
     54 
     55     float3 mom ;
     56     float  weight ;
     57 
     58     float3 pol ;
     59     float  wavelength ;
     60 
     61     unsigned boundary_flag ;  // p.view(np.uint32)[3,0] 
     62     unsigned identity ;       // p.view(np.uint32)[3,1]
     63     unsigned orient_idx ;     // p.view(np.uint32)[3,2]
     64     unsigned flagmask ;       // p.view(np.uint32)[3,3]
     65 
     66 
     67     SPHOTON_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient );


     72     SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it 
     73     SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 
     74 
    105 SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_ )
    106 {
    107     set_boundary(boundary_);
    108     identity = identity_ ;
    109     set_orient( orient_ );
    110 }





Hmm, I think the 37884 is a global iidx across all solid. Not the same as the MOI iidx ?
The maximum iidx that midx 117 stretches to is 12614::

    epsilon:CSG blyth$ MOI=117:0:12614 ./CSGTargetTest.sh remote 
     moi     117:0:12614 midx   117 mord     0 iidx  12614 name NNVTMCPPMTsMask_virtual0x5f5f0e0
     moi     117:0:12614 ce (-1914.582,219.678,-19332.773,309.724) 
     moi     117:0:12614 q0 (-0.989, 0.113, 0.099, 0.000) (-0.114,-0.993, 0.000, 0.000) ( 0.099,-0.011, 0.995, 0.000) (-1915.115,219.739,-19338.160, 1.000) 


::

    epsilon:CSG blyth$ METH=descInstance IDX=37684,0,48476,48477 ./CSGTargetTest.sh remote
    CSGFoundry::descInstance idx   37684 inst.size   48477 idx   37684 ins 37684 gas  2 ias 0 so CSGSolid               r2 primNum/Offset     7 3094 ce ( 0.000, 0.000, 0.025,264.050) 
    CSGFoundry::descInstance idx       0 inst.size   48477 idx       0 ins     0 gas  0 ias 0 so CSGSolid               r0 primNum/Offset  3089    0 ce ( 0.000, 0.000, 0.000,60000.000) 
    CSGFoundry::descInstance idx   48476 inst.size   48477 idx   48476 ins 48476 gas  9 ias 0 so CSGSolid               r9 primNum/Offset   130 3118 ce ( 0.000, 0.000, 0.000,3430.600) 
    CSGFoundry::descInstance idx   48477 inst.size   48477 idx OUT OF RANGE 


::

    epsilon:tests blyth$ INST=37684 CSGFoundry_getFrame_Test
     INST 37684
     fr 
     frs -
     ce  ( 0.000, 0.000, 0.025,264.050) 
     m2w ( 0.155, 0.890, 0.429, 0.000) (-0.985, 0.171,-0.000, 0.000) (-0.074,-0.423, 0.903, 0.000) (1430.869,8223.110,-17550.311, 1.000) 
     w2m ( 0.155,-0.985,-0.074, 0.000) ( 0.890, 0.171,-0.423, 0.000) ( 0.429,-0.000, 0.903, 0.000) ( 0.009,-0.005,19434.000, 1.000) 
     midx    0 mord    0 iidx    0
     inst 37684
     ix0     0 ix1     0 iy0     0 iy1     0 iz0     0 iz1     0 num_photon    0
     ins  37684 gas     2 ias     0


    descInstance
    CSGFoundry::descInstance idx   37684 inst.size   48477 idx   37684 ins 37684 gas  2 ias 0 so CSGSolid               r2 primNum/Offset     7 3094 ce ( 0.000, 0.000, 0.025,264.050) 



Step record end positions in ballpark of INST:37684::

    In [5]: x.record[0][-4:]                                                                                                                                                                                  
    Out[5]: 
    array([[[  1517.433,   8343.771, -17343.04 ,    183.436],
            [    -0.593,      0.349,     -0.726,      1.   ],
            [     0.195,     -0.812,     -0.55 ,    500.   ],
            [     0.   ,      0.   ,     -0.   ,      0.   ]],

           [[  1517.432,   8343.771, -17343.04 ,    183.436],
            [    -0.593,      0.349,     -0.726,      1.   ],
            [     0.195,     -0.812,     -0.55 ,    500.   ],
            [     0.   ,      0.   ,     -0.   ,      0.   ]],

           [[  1517.432,   8343.771, -17343.04 ,    183.436],
            [    -0.593,      0.349,     -0.726,      1.   ],
            [     0.195,     -0.812,     -0.55 ,    500.   ],
            [     0.   ,      0.   ,     -0.   ,      0.   ]],

           [[  1517.432,   8343.771, -17343.04 ,    183.436],
            [    -0.593,      0.349,     -0.726,      1.   ],
            [     0.195,     -0.812,     -0.55 ,    500.   ],
            [     0.   ,      0.   ,     -0.   ,      0.   ]]], dtype=float32)









Scaling up step-to-step diffs shows have sequence of micro steps of 0.000244 or 0.000122 mm::

    In [16]: 1e3*(r[0,1:,:3] - r[0,:-1,:3])                                                                                                                                                                   
    Out[16]: 
    array([[[ 13355625.   ,         0.   ,         0.   ,     68512.27 ],
            [    -1517.013,       371.099,      -771.352,         0.   ],
            [     -206.617,     -1928.593,      -308.26 ,         0.   ]],

           [[-10556294.   ,   7577046.   , -15749353.   ,    104740.45 ],
            [        3.604,         0.939,        -1.952,         0.   ],
            [      413.758,       107.832,      -224.135,         0.   ]],

           [[   -82892.58 ,     60067.383,   -124854.49 ,       825.592],
            [      -79.139,       -22.808,        47.408,         0.   ],
            [      -11.869,         8.778,       -17.653,         0.   ]],

           [[ -1199005.6  ,    706658.2  ,  -1468832.   ,      9357.27 ],
            [        0.   ,         0.   ,         0.   ,         0.   ],
            [       -0.   ,         0.   ,         0.   ,         0.   ]],

           [[       -0.244,         0.   ,         0.   ,         0.   ],
            [        0.   ,         0.   ,         0.   ,         0.   ],
            [        0.   ,        -0.   ,        -0.   ,         0.   ]],

           [[       -0.244,         0.   ,         0.   ,         0.   ],
            [        0.   ,         0.   ,         0.   ,         0.   ],
            [       -0.   ,         0.   ,         0.   ,         0.   ]],

           [[       -0.122,         0.   ,         0.   ,         0.   ],
            [        0.   ,         0.   ,         0.   ,         0.   ],
            [        0.   ,        -0.   ,         0.   ,         0.   ]],

           [[       -0.122,         0.   ,         0.   ,         0.   ],
            [        0.   ,         0.   ,         0.   ,         0.   ],
            [       -0.   ,         0.   ,         0.   ,         0.   ]],

           [[       -0.122,         0.   ,         0.   ,         0.   ],
            [        0.   ,         0.   ,         0.   ,         0.   ],
            [       -0.   ,         0.   ,         0.   ,         0.   ]]], dtype=float32)




Take a look at bnd:27::

    epsilon:CSG blyth$ ./CSGPrimTest.sh remote | grep bnd:27
      pri:3085  lpr:3085   gas:0 msh:126  bnd:27   nno:1 nod:23199 ce (      0.00,      0.00,  19787.00,   1963.00) meshName sWaterTube0x71a5330 bndName   Water///Water
      pri:3089     lpr:0   gas:1 msh:122  bnd:27   nno:3 nod:23207 ce (      0.00,      0.00,    -17.94,     57.94) meshName PMT_3inch_pmt_solid0x66e51d0 bndName   Water///Water
      pri:3094     lpr:0   gas:2 msh:117  bnd:27   nno:7 nod:23214 ce (      0.00,      0.00,      5.41,    264.05) meshName NNVTMCPPMTsMask_virtual0x5f5f0e0 bndName   Water///Water
      pri:3101     lpr:0   gas:3 msh:110  bnd:27   nno:7 nod:23247 ce (      0.00,      0.00,      8.41,    264.05) meshName HamamatsuR12860sMask_virtual0x5f50520 bndName   Water///Water
    epsilon:CSG blyth$ 




HMM : would be good to see a simtrace oriented in this region 
