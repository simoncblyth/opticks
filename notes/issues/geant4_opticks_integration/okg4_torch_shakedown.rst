OKG4 Torch Shakedown
======================



Simulation
-----------

::

   OKG4Test --compute --save

   OKG4Test --compute --save --steppingdbg     ## very verbose


   lldb OKG4Test -- --compute --save 

   (lldb) b "OpRayleigh::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)" 


Analysis
----------

::

   tokg4.py 

   ipython -i $(which tokg4.py)


Viz
----

::

    OKTest --load            # works

    OKTest --load --vizg4    # was failing wrt photon buffer

    OKG4Test --load --vizg4   # succeeds to load g4evt, fix index loading with indexPresentationPrep, but suspect using Opticks index with G4 evt 


vizg4 using opticks index ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OKTest vizg4 fail
~~~~~~~~~~~~~~~~~~~~

This was using old propagator upload, fixed by move to viz->uploadEvent following OKG4Test::

    2016-09-30 20:44:59.861 INFO  [1132529] [OpticksViz::uploadEvent@281] OpticksViz::uploadEvent (0) DONE 
    2016-09-30 20:44:59.861 INFO  [1132529] [OEvent::createBuffers@62] OEvent::createBuffers  genstep 1,6,4 nopstep NULL photon 100000,4,4 record 100000,10,2,4 phosel 100000,1,4 recsel 100000,10,1,4 sequence 100000,1,2 seed 0,1,1 hit 897,4,4
    2016-09-30 20:44:59.861 FATAL [1132529] [OContext::createBuffer@423] OContext::createBuffer CANNOT createBufferFromGLBO as not uploaded   name               photon buffer_id -1
    Assertion failed: (buffer_id > -1), function createBuffer, file /Users/blyth/opticks/optixrap/OContext.cc, line 427.
    Abort trap: 6

OKG4Test vizg4 index fail
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2016-09-30 20:47:25.236 INFO  [1133593] [OpticksEvent::importParameters@1593] OpticksEvent::importParameters  mode_ COMPUTE_MODE --> COMPUTE_MODE
    2016-09-30 20:47:25.238 WARN  [1133593] [*Index::load@370] Index::load FAILED to load index  idpath /tmp/blyth/opticks/evt/dayabay/torch/-1 itemtype Boundary_Index Source path /tmp/blyth/opticks/evt/dayabay/torch/-1/Boundary_IndexSource.json Local path /tmp/blyth/opticks/evt/dayabay/torch/-1/Boundary_IndexLocal.json



Initially not in same ballpark, after an afternoon get into same ballpark
----------------------------------------------------------------------------

This is torch running with a point source and big bugs, 
position and polarization were wrong ::

       A:seqhis_ana    1:dayabay 
              8ccccd        0.434          43405       [6 ] TO BT BT BT BT SA
          ccaccccccd        0.090           9009       [10] TO BT BT BT BT BT BT SR BT BT
             4cccccd        0.061           6104       [7 ] TO BT BT BT BT BT AB
                  4d        0.061           6051       [2 ] TO AB
          cccbcccccd        0.038           3822       [10] TO BT BT BT BT BT BR BT BT BT
             8cccccd        0.030           2978       [7 ] TO BT BT BT BT BT SA
           8cbcccccd        0.025           2511       [9 ] TO BT BT BT BT BT BR BT SA
             8cccc6d        0.022           2165       [7 ] TO SC BT BT BT BT SA
            8ccccccd        0.016           1565       [8 ] TO BT BT BT BT BT BT SA
                4ccd        0.013           1347       [4 ] TO BT BT AB
          cccc9ccccd        0.013           1326       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc5d        0.011           1126       [7 ] TO RE BT BT BT BT SA
          cccacccccd        0.011           1103       [10] TO BT BT BT BT BT SR BT BT BT
          accccccccd        0.009            861       [10] TO BT BT BT BT BT BT BT BT SR
                 45d        0.008            833       [3 ] TO RE AB
              4ccccd        0.007            701       [6 ] TO BT BT BT BT AB
             7cccccd        0.006            636       [7 ] TO BT BT BT BT BT SD
            8cccc55d        0.006            609       [8 ] TO RE RE BT BT BT BT SA
           4cccccccd        0.005            542       [9 ] TO BT BT BT BT BT BT BT AB
          cccccccccd        0.004            450       [10] TO BT BT BT BT BT BT BT BT BT
                          100000         1.00 
       B:seqhis_ana   -1:dayabay 
          666666666d        0.141          14077       [10] TO SC SC SC SC SC SC SC SC SC
                  4d        0.084           8431       [2 ] TO AB
                 46d        0.072           7202       [3 ] TO SC AB
                  8d        0.070           6966       [2 ] TO SA
                866d        0.065           6510       [4 ] TO SC SC SA
               8666d        0.063           6344       [5 ] TO SC SC SC SA
                 86d        0.061           6118       [3 ] TO SC SA
                466d        0.059           5900       [4 ] TO SC SC AB
              86666d        0.056           5624       [6 ] TO SC SC SC SC SA
             866666d        0.051           5056       [7 ] TO SC SC SC SC SC SA
               4666d        0.049           4887       [5 ] TO SC SC SC AB
            8666666d        0.043           4257       [8 ] TO SC SC SC SC SC SC SA
              46666d        0.038           3785       [6 ] TO SC SC SC SC AB
           86666666d        0.034           3375       [9 ] TO SC SC SC SC SC SC SC SA
             466666d        0.030           2979       [7 ] TO SC SC SC SC SC AB
          866666666d        0.028           2802       [10] TO SC SC SC SC SC SC SC SC SA
            4666666d        0.024           2412       [8 ] TO SC SC SC SC SC SC AB
           46666666d        0.018           1847       [9 ] TO SC SC SC SC SC SC SC AB
          466666666d        0.014           1381       [10] TO SC SC SC SC SC SC SC SC AB
          666666cc6d        0.000              4       [10] TO SC BT BT SC SC SC SC SC SC


Now at least in same ballpark, some zero flags to identify::

     A:seqhis_ana    1:dayabay 
              8ccccd        0.434          43405       [6 ] TO BT BT BT BT SA
          ccaccccccd        0.090           9009       [10] TO BT BT BT BT BT BT SR BT BT
             4cccccd        0.061           6104       [7 ] TO BT BT BT BT BT AB
                  4d        0.061           6051       [2 ] TO AB
          cccbcccccd        0.038           3822       [10] TO BT BT BT BT BT BR BT BT BT
             8cccccd        0.030           2978       [7 ] TO BT BT BT BT BT SA
           8cbcccccd        0.025           2511       [9 ] TO BT BT BT BT BT BR BT SA
             8cccc6d        0.022           2165       [7 ] TO SC BT BT BT BT SA
            8ccccccd        0.016           1565       [8 ] TO BT BT BT BT BT BT SA
                4ccd        0.013           1347       [4 ] TO BT BT AB
          cccc9ccccd        0.013           1326       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc5d        0.011           1126       [7 ] TO RE BT BT BT BT SA
          cccacccccd        0.011           1103       [10] TO BT BT BT BT BT SR BT BT BT
          accccccccd        0.009            861       [10] TO BT BT BT BT BT BT BT BT SR
                 45d        0.008            833       [3 ] TO RE AB
              4ccccd        0.007            701       [6 ] TO BT BT BT BT AB
             7cccccd        0.006            636       [7 ] TO BT BT BT BT BT SD
            8cccc55d        0.006            609       [8 ] TO RE RE BT BT BT BT SA
           4cccccccd        0.005            542       [9 ] TO BT BT BT BT BT BT BT AB
          cccccccccd        0.004            450       [10] TO BT BT BT BT BT BT BT BT BT
                          100000         1.00 
       B:seqhis_ana   -1:dayabay 
            8ccccccd        0.423          42297       [8 ] TO BT BT BT BT BT BT SA
          cc0ccccccd        0.115          11515       [10] TO BT BT BT BT BT BT ?0? BT BT
                  4d        0.076           7584       [2 ] TO AB
          c0c0c0cccd        0.067           6652       [10] TO BT BT BT ?0? BT ?0? BT ?0? BT
          cccbcccccd        0.062           6219       [10] TO BT BT BT BT BT BR BT BT BT
          cccccccccd        0.031           3065       [10] TO BT BT BT BT BT BT BT BT BT
          ccbccccccd        0.023           2309       [10] TO BT BT BT BT BT BT BR BT BT
                4ccd        0.019           1902       [4 ] TO BT BT AB
           8cccccc6d        0.016           1552       [9 ] TO SC BT BT BT BT BT BT SA
          c0c00cc0cd        0.013           1325       [10] TO BT ?0? BT BT ?0? ?0? BT ?0? BT
              8ccccd        0.012           1167       [6 ] TO BT BT BT BT SA
           b0ccccccd        0.008            815       [9 ] TO BT BT BT BT BT BT ?0? BR
           8cbcccccd        0.007            681       [9 ] TO BT BT BT BT BT BR BT SA
              4ccccd        0.007            674       [6 ] TO BT BT BT BT AB
             4cccccd        0.006            570       [7 ] TO BT BT BT BT BT AB
          ccc0b0cccd        0.005            481       [10] TO BT BT BT ?0? BR ?0? BT BT BT
            4ccccccd        0.005            480       [8 ] TO BT BT BT BT BT BT AB
          c0cccccc6d        0.004            440       [10] TO SC BT BT BT BT BT BT ?0? BT
          c0b0c0cccd        0.004            385       [10] TO BT BT BT ?0? BT ?0? BR ?0? BT
          cbcccccccd        0.004            384       [10] TO BT BT BT BT BT BT BT BR BT


After identify SR SURFACE_SREFLECT with SpikeReflection eliminate some zero flags,
suspect remainder due to SameMaterial steps::
 
       B:seqhis_ana   -1:dayabay 
            8ccccccd        0.420         419905       [8 ] TO BT BT BT BT BT BT SA      
          ccaccccccd        0.081          81049       [10] TO BT BT BT BT BT BT SR BT BT
                  4d        0.078          77610       [2 ] TO AB
          c0cac0cccd        0.066          66482       [10] TO BT BT BT ?0? BT SR BT ?0? BT
          cccbcccccd        0.063          63079       [10] TO BT BT BT BT BT BR BT BT BT
          cc9ccccccd        0.034          33940       [10] TO BT BT BT BT BT BT DR BT BT
          cccccccccd        0.031          30534       [10] TO BT BT BT BT BT BT BT BT BT
          ccbccccccd        0.023          23444       [10] TO BT BT BT BT BT BT BR BT BT
                4ccd        0.019          19127       [4 ] TO BT BT AB
           8cccccc6d        0.015          15140       [9 ] TO SC BT BT BT BT BT BT SA
          cac00cc0cd        0.013          12771       [10] TO BT ?0? BT BT ?0? ?0? BT SR BT
              8ccccd        0.012          12083       [6 ] TO BT BT BT BT SA
          abaccccccd        0.008           8032       [10] TO BT BT BT BT BT BT SR BR SR
           8cbcccccd        0.008           7512       [9 ] TO BT BT BT BT BT BR BT SA
              4ccccd        0.007           7050       [6 ] TO BT BT BT BT AB
             4cccccd        0.006           5645       [7 ] TO BT BT BT BT BT AB
          ccc0b0cccd        0.005           4873       [10] TO BT BT BT ?0? BR ?0? BT BT BT
            4ccccccd        0.005           4847       [8 ] TO BT BT BT BT BT BT AB
          cbcccccccd        0.004           3735       [10] TO BT BT BT BT BT BT BT BR BT
          cabac0cccd        0.004           3725       [10] TO BT BT BT ?0? BT SR BR SR BT
                         1000000         1.00 



zero flags, SR?
~~~~~~~~~~~~~~~~~

* SR : SURFACE_SREFLECT is specular reflection, which is not mirrored in CG4 yet 


::

    simon:~ blyth$ find /usr/local/opticks -name abbrev.json
    /usr/local/opticks/opticksdata/export/DayaBay/GMaterialLib/abbrev.json
    /usr/local/opticks/opticksdata/resource/GFlags/abbrev.json
    simon:~ blyth$ cat /usr/local/opticks/opticksdata/resource/GFlags/abbrev.json
    {
        "CERENKOV":"CK",
        "SCINTILLATION":"SI",
        "TORCH":"TO",
        "MISS":"MI",
        "BULK_ABSORB":"AB",
        "BULK_REEMIT":"RE", 
        "BULK_SCATTER":"SC",    
        "SURFACE_DETECT":"SD",
        "SURFACE_ABSORB":"SA",      
        "SURFACE_DREFLECT":"DR",
        "SURFACE_SREFLECT":"SR",
        "BOUNDARY_REFLECT":"BR",
        "BOUNDARY_TRANSMIT":"BT",
        "NAN_ABORT":"NA"
    }





FIXED : Positional bug
----------------------------------------------

Photons should all be starting from same place::

    ipython -i $(which tokg4.py)

    In [2]: a.rpost_(0)
    Out[2]: 
    A()sliced
    A([[ -18079.4443, -799699.4149,   -6604.9499,       0.0977],
           [ -18079.4443, -799699.4149,   -6604.9499,       0.0977],
           [ -18079.4443, -799699.4149,   -6604.9499,       0.0977],
           ..., 
           [ -18079.4443, -799699.4149,   -6604.9499,       0.0977],
           [ -18079.4443, -799699.4149,   -6604.9499,       0.0977],
           [ -18079.4443, -799699.4149,   -6604.9499,       0.0977]])

    In [2]: a.gs
    Out[2]: 
    A(torch,1,dayabay)-
    A([[[      0.    ,       0.    ,       0.    ,       0.    ],
            [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.    ,       0.    ,       1.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,     430.    ],
            [      0.    ,       1.    ,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]]], dtype=float32)



    ## huh: B photons not starting from where genstep points
    ## OR    CTorchSource::configure _t 0.1 _radius 0 _pos -18079.4531,-799699.4375,-6605.0000 


    In [3]: b.rpost_(0)
    Out[3]: 
    A()sliced
    A([[ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           ..., 
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977]])

    In [3]: b.gs
    Out[3]: 
    A(torch,-1,dayabay)-
    A([[[      0.    ,       0.    ,       0.    ,       0.    ],
            [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.    ,       0.    ,       1.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,     430.    ],
            [      0.    ,       1.    ,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]]], dtype=float32)



    ## primaries collected from CTorchSource all at (0,0,0,0,0.1)
    ## where is frame setup for the default torch source done ?

    In [4]: pr = np.load("/tmp/blyth/opticks/cg4/primary.npy")

    In [5]: pr
    Out[5]: 
    array([[[ 0. ,  0. ,  0. ,  0.1],
            [ 0. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  0. ]],



The gensteps have position that looks to be frame targetted::

     21 const char* TorchStepNPY::DEFAULT_CONFIG =
     22     "type=sphere_"
     23     "frame=3153_"
     24     "source=0,0,0_"
     25     "target=0,0,1_"
     26     "photons=100000_"
     27     "material=GdDopedLS_"
     28     "wavelength=430_"
     29     "weight=1.0_"
     30     "time=0.1_"
     31     "zenithazimuth=0,1,0,1_"
     32     "radius=0_" ;
     33 
     34 //  Aug 2016: change default torch wavelength from 380nm to 430nm
     35 //
     36 //
     37 // NB time 0.f causes 1st step record rendering to be omitted, as zero is special
     38 // NB the material string needs to be externally translated into a material line



Huh CTorchSource operating direct from TorchStepNPY, not the targetted NPY that it creates::

     35 
     36 CTorchSource::CTorchSource(TorchStepNPY* torch, unsigned int verbosity)
     37     :
     38     CSource(verbosity),
     39     m_torch(torch),


* does this mean that missed the targetting 



bouncemax zero check
------------------------

With bouncemax zero propagation is immediately terminated in both Opticks and G4, 
so can see initial photon position from photon buffer
without the compression/decompression complications of the record buffer::

    OKG4Test --save --compute --bouncemax 0

::

    In [5]: a.ox[:,0]   ## Opticks as expected
    Out[5]: 
    A()sliced
    A([[ -18079.453, -799699.438,   -6605.   ,       0.1  ],
           [ -18079.453, -799699.438,   -6605.   ,       0.1  ],
           [ -18079.453, -799699.438,   -6605.   ,       0.1  ],
           ..., 
           [ -18079.453, -799699.438,   -6605.   ,       0.1  ],
           [ -18079.453, -799699.438,   -6605.   ,       0.1  ],
           [ -18079.453, -799699.438,   -6605.   ,       0.1  ]], dtype=float32)



    In [6]: b.ox[:,0]    ## G4: real crazy position and time 
    Out[6]: 
    A()sliced
    A([[       0.   ,        0.   , -2400000.   ,     8005.638],
           [       0.   ,        0.   ,  -816713.875,     2724.364],
           [       0.   ,        0.   , -1618713.875,     5399.548],
           ..., 
           [       0.   ,        0.   , -2062325.125,     6879.276],
           [       0.   ,        0.   , -2400000.   ,     8005.638],
           [       0.   ,        0.   , -1681468.25 ,     5608.874]], dtype=float32)



After handling sphere positioning, gets a bit better::

    In [2]: a.ox[:,0]
    Out[2]: 
    A()sliced
    A([[ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
           [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
           [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
           ..., 
           [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
           [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
           [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ]], dtype=float32)

    In [3]: b.ox[:,0]
    Out[3]: 
    A()sliced
    A([[ -18079.4531, -799699.4375,   -8635.    ,      10.5231],
           [ -18079.4531, -799699.4375,   -6798.9727,       1.096 ],
           [ -18079.4531, -799699.4375,   -8635.    ,      10.5231],
           ..., 
           [ -18079.4531, -799699.4375,   -8635.    ,      10.5231],
           [ -18079.4531, -799699.4375,   -8635.    ,      10.5231],
           [ -18079.4531, -799699.4375,   -8635.    ,      10.5231]], dtype=float32)



Direction should be random not all in -z dir::


    In [8]: pr = np.load("cg4/primary.npy")

    In [9]: pr
    Out[9]: 
    array([[[ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.    ,       0.    ,      -1.    ,       1.    ],
            [      1.    ,       0.    ,       0.    ,     430.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]],


After setting **iso** get::

    In [10]: pr = np.load("cg4/primary.npy")

    In [11]: pr
    Out[11]: 
    array([[[ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.6034,      -0.673 ,      -0.4279,       1.    ],
            [      0.7975,       0.5092,       0.3237,     430.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]],

           [[ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.084 ,      -0.4561,       0.886 ,       1.    ],
            [      0.9965,       0.0384,      -0.0747,     430.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]],

           [[ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [     -0.2997,      -0.7136,      -0.6332,       1.    ],
            [      0.954 ,      -0.2242,      -0.1989,     430.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]],





Material reporting not operational in CG4
--------------------------------------------

::

      A:seqmat_ana    1:dayabay 
              443231        0.441          44062       [6 ] Gd Ac LS Ac MO MO
          33ff343231        0.090           9021       [10] Gd Ac LS Ac MO Ac Ai Ai Ac Ac
                  11        0.061           6051       [2 ] Gd Gd
             aa33231        0.049           4859       [7 ] Gd Ac LS Ac Ac ES ES
          3343343231        0.037           3688       [10] Gd Ac LS Ac MO Ac Ac MO Ac Ac
             4432311        0.034           3351       [7 ] Gd Gd Ac LS Ac MO MO
             dd43231        0.032           3164       [7 ] Gd Ac LS Ac MO Vm Vm
           443343231        0.021           2116       [9 ] Gd Ac LS Ac MO Ac Ac MO MO
          3323443231        0.015           1465       [10] Gd Ac LS Ac MO MO Ac LS Ac Ac
                2231        0.013           1319       [4 ] Gd Ac LS LS
             aa34231        0.011           1104       [7 ] Gd Ac LS MO Ac ES ES
                 111        0.011           1067       [3 ] Gd Gd Gd
             4443231        0.009            940       [7 ] Gd Ac LS Ac MO MO MO
          ff33424321        0.008            832       [10] Gd LS Ac MO LS MO Ac Ac Ai Ai
            44323111        0.008            780       [8 ] Gd Gd Gd Ac LS Ac MO MO
            dde43231        0.007            693       [8 ] Gd Ac LS Ac MO Py Vm Vm
          334ff33231        0.007            656       [10] Gd Ac LS Ac Ac Ai Ai MO Ac Ac
             4432231        0.006            559       [7 ] Gd Ac LS LS Ac MO MO
           44ee43231        0.005            465       [9 ] Gd Ac LS Ac MO Py Py MO MO
            44343231        0.004            423       [8 ] Gd Ac LS Ac MO Ac MO MO
                          100000         1.00 
       B:seqmat_ana   -1:dayabay 
            11111111        0.429          42900       [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
          1111111111        0.399          39924       [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
                  11        0.076           7584       [2 ] Gd Gd
           111111111        0.037           3712       [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd
                1111        0.020           1952       [4 ] Gd Gd Gd Gd
              111111        0.019           1900       [6 ] Gd Gd Gd Gd Gd Gd
             1111111        0.012           1208       [7 ] Gd Gd Gd Gd Gd Gd Gd
                 111        0.005            455       [3 ] Gd Gd Gd
               11111        0.004            365       [5 ] Gd Gd Gd Gd Gd
                          100000         1.00 



