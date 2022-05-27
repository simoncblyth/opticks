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


* reviewing where this identity info comes from in :doc:`identity-review`


Hmm, I think the 37884 is a global iidx across all solid. Not the same as the MOI iidx ?  

* Yes: it is the instanceId off the geometry (__closesthit__ch) passed via PRD quad2


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



Added flat inst_idx interpretation of MOI to CSGFoundry, so can target using the flat inst_idx::

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




HMM : would be good to see a simtrace in this region 
-------------------------------------------------------------

* see :doc:`simtrace-shakedown`


::

   cx 
   ./cxs_debug.sh 


    epsilon:CSGOptiX blyth$ cat cxs_debug.sh 
    #!/bin/bash -l 

    moi=37684
    ce_offset=0,-64.59664,0    # -Y shift aligning slice plane with a cxsim photon 0 hit with microsteps 
    ce_scale=1   
    cegs=16:0:9:500   
    gridscale=0.10

    export ZOOM=2
    export LOOK=209.774,-64.59664,129.752

    source ./cxs.sh $*



The microsteps are very close to::

      0 : 3094 :  46346 :                  red :         NNVTMCPPMTsMask_virtual0x5f5f0e0 : NNVTMCPPMTsMask_virtual0x5f5f0e0  

 
That solid looks like a doubled slightly offset surface ?




