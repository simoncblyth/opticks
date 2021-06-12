CRecorder_duplicated_point_BTBT
===================================


issue
-------

* Geant4/CRecorder giving an extra BT cf Opticks from a very close set of boundaries
* such a different history needs to be fixed for the Opticks matching machinery to work 

::

    als[:10]
    TO BT BT AB
    TO BT BT BT SD
    TO BT BT BT SA
    TO BT BT AB
    TO BT BT BT SD
    TO BT BT BT SD
    TO BT BT BT SA
    TO AB

    bls[:10]
    TO BT BT BT *BT* SA
    TO SC BT BT *BT* SA
    TO BT BT BT *BT* SD
    TO AB
    TO SC SC BT BT BT *BT* SD
    TO BT BT BT *BT* SA
    TO BT BT AB
    TO SC BT BT BR SA


tds3ip using --dbgseqhis


::

    tds3ip () 
    { 
        local name="CubeCorners";
        local path="$HOME/.opticks/InputPhotons/${name}.npy";
        export OPTICKS_EVENT_PFX=tds3ip;
        export INPUT_PHOTON_PATH=$path;
        tds3 --dbgseqhis 0x7ccccd
    }



    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump@188] CDebug::postTrack
    2021-06-11 22:06:36.140 INFO  [446353] [CRec::dump@194] CDebug::dump record_id 5  origin[ 0.577-0.5770.577]   Ori[ 0.577-0.5770.577] 
    2021-06-11 22:06:36.140 INFO  [446353] [CRec::dump@200]  nstp 5
    (0 )  TO/BT     FrT                       PRE_SAVE POST_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre                   pTarget              LS          noProc           Undefined pos[      0.000     0.000     0.000]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns  0.600 nm 440.000 mm/ns 195.234
     post                  pAcrylic         Acrylic  Transportation        GeomBoundary pos[  10218.522-10218.522 10218.522]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns 91.255 nm 440.000 mm/ns 193.809
     )
    (1 )  BT/BT     FrT                                           POST_SAVE 
    [   1](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre                  pAcrylic         Acrylic  Transportation        GeomBoundary pos[  10218.522-10218.522 10218.522]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns 91.255 nm 440.000 mm/ns 193.809
     post               pInnerWater           Water  Transportation        GeomBoundary pos[  10287.804-10287.804 10287.804]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns 91.875 nm 440.000 mm/ns 216.910
     )
    (2 )  BT/BT     FrT                                           POST_SAVE 
    [   2](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre               pInnerWater           Water  Transportation        GeomBoundary pos[  10287.804-10287.804 10287.804]  dir[    0.577  -0.577   0.577]  pol[   -0.707   0.000   0.707]  ns 91.875 nm 440.000 mm/ns 216.910
     post         pLPMT_NNVT_MCPPMT           Pyrex  Transportation        GeomBoundary pos[  11139.935-11139.935 11139.935]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.679 nm 440.000 mm/ns 196.979
     )
    (3 )  BT/BT     SAM                                           POST_SAVE 
    [   3](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre         pLPMT_NNVT_MCPPMT           Pyrex  Transportation        GeomBoundary pos[  11139.935-11139.935 11139.935]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.679 nm 440.000 mm/ns 196.979
     post      NNVTMCPPMT_body_phys           Pyrex  Transportation        GeomBoundary pos[  11139.936-11139.936 11139.936]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.679 nm 440.000 mm/ns 196.979
     )
    (4 )  BT/SD     Det              POST_SAVE POST_DONE LAST_POST SURF_ABS 
    [   4](Stp ;opticalphoton stepNum    5(tk ;opticalphoton tid 6 pid 0 nm    440 mm  ori[    0.577  -0.577   0.577]  pos[ 11143.476-11143.09611143.087]  )
      pre      NNVTMCPPMT_body_phys           Pyrex  Transportation        GeomBoundary pos[  11139.936-11139.936 11139.936]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.679 nm 440.000 mm/ns 196.979
     post    NNVTMCPPMT_inner1_phys          Vacuum  Transportation        GeomBoundary pos[  11143.476-11143.096 11143.087]  dir[    0.621  -0.555   0.553]  pol[    0.695   0.717  -0.061]  ns 98.708 nm 440.000 mm/ns 196.979
     )
    2021-06-11 22:06:36.140 INFO  [446353] [CRec::dump@204]  npoi 0
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_brief@204] CRecorder::dump_brief m_ctx._record_id        5 m_photon._badflag     0 --dbgseqhis  sas: POST_SAVE POST_DONE LAST_POST SURF_ABS 
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_brief@213]  seqhis           7ccccd    TO BT BT BT BT SD                               
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_brief@218]  mskhis             1840    SD|BT|TO
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_brief@223]  seqmat           deefb1    LS Acrylic Water Pyrex Pyrex Vacuum - - - - - - - - - - 
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_sequence@231] CDebug::dump_sequence
    2021-06-11 22:06:36.140 INFO  [446353] [CDebug::dump_points@257] CDeug::dump_points



Hmm why not exactly matching the above ? Must be going thru a float ? YEP move to `setQuad_`::

    In [3]: b.dx[5]                                                                                                                                                                                          
    Out[3]: 
    A([[[     0.5774,     -0.5774,      0.5774,      0.6   ],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 10219.0996, -10219.0996,  10219.0996,     91.2555],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 10288.3818, -10288.3818,  10288.3818,     91.8747],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 11140.5127, -11140.5127,  11140.5127,     98.679 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[ 11140.5127, -11140.5127,  11140.5127,     98.679 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[ 11144.0537, -11143.6738,  11143.6641,     98.708 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]],




Notice for m_stp index 3 

* pre and post points are at same position, BUT are in different volumes : pLPMT_NNVT_MCPPMT, NNVTMCPPMT_body_phys
* same-ness was because of accidently getting narrowed to float precision on the way into deluxe buffer
* G4OpBoundaryProcessStatus SAM is abbrev for SameMaterial


tds3ip.sh 1::

   In [2]: 10218.522*math.sqrt(3)                                                                                                                                                                           
   Out[2]: 17698.999282260338

    In [3]: b.dx.shape                                                                                                                                                                                       
    Out[3]: (8, 10, 2, 4)

    In [4]: b.dx[5]                                                                                                                                                                                          
    Out[4]: 
    A([[[     0.5774,     -0.5774,      0.5774,      0.6   ],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 10219.0996, -10219.0996,  10219.0996,     91.2555],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 10288.3818, -10288.3818,  10288.3818,     91.8747],
        [    -0.7071,      0.    ,      0.7071,    440.    ]],

       [[ 11140.5127, -11140.5127,  11140.5127,     98.679 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[ 11140.5127, -11140.5127,  11140.5127,     98.679 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[ 11144.0537, -11143.6738,  11143.6641,     98.708 ],
        [     0.6946,      0.7168,     -0.0615,    440.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]],

       [[     0.    ,      0.    ,      0.    ,      0.    ],
        [     0.    ,      0.    ,      0.    ,      0.    ]]])



