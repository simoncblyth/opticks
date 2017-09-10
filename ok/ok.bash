ok-src(){      echo ok/ok.bash ; }
ok-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ok-src)} ; }
ok-vi(){       vi $(ok-source) ; }

ok-usage(){ cat << EOU

OK : Formerly GGeoView
=========================

Start from glfwtest- and add in OptiX functionality from optixrap-

* NB raytrace- is another user of optixwrap- 


See also
----------

* ggv- examples of ggeoview invokations




OTracerTest and ggeoview running FAILING with malloc errors
--------------------------------------------------------------

Have observed the below before, usually after a long while away. 
Symptom is malloc errors related to setting up the optix context that go away 
without obvious reason after a little investigation and recompilations.  

Running without optix works::

    op --nooptix

Perhaps a stale ptx build issue ? 


::

    simon:env blyth$ op --tracer

    save dir /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae name  GFlagsLocal.ini
    OTracerTest(83607,0x7fff74d63310) malloc: *** error for object 0x7fdd0bdaa408: incorrect checksum for freed object - object was probably modified after being freed.
    *** set a breakpoint in malloc_error_break to debug
    /Users/blyth/env/bin/op.sh: line 374: 83607 Abort trap: 6           /usr/local/opticks/bin/OTracerTest --tracer


::

        frame #11: 0x0000000102919d6d liboptix.1.dylib`___lldb_unnamed_function3395$$liboptix.1.dylib + 173
        frame #12: 0x0000000102765e86 liboptix.1.dylib`___lldb_unnamed_function1141$$liboptix.1.dylib + 678
        frame #13: 0x000000010268e04a liboptix.1.dylib`rtBufferMap + 122
        frame #14: 0x00000001035fd6cf libOptiXRap.dylib`optix::BufferObj::map(this=0x00000001165321c0) + 47 at optixpp_namespace.h:3755
        frame #15: 0x0000000103619647 libOptiXRap.dylib`OPropertyLib::makeTexture(this=0x0000000116532240, buffer=0x0000000112f58a70, format=RT_FORMAT_FLOAT, nx=1024, ny=1, empty=false) + 871 at OPropertyLib.cc:38
        frame #16: 0x0000000103618f38 libOptiXRap.dylib`OSourceLib::makeSourceTexture(this=0x0000000116532240, buf=0x0000000112f58a70) + 824 at OSourceLib.cc:34
        frame #17: 0x0000000103618bd9 libOptiXRap.dylib`OSourceLib::convert(this=0x0000000116532240) + 313 at OSourceLib.cc:12
        frame #18: 0x000000010450287e libOpticksOp.dylib`OpEngine::prepareOptiX(this=0x000000010650a300) + 4606 at OpEngine.cc:87
        frame #19: 0x000000010465aebe libGGeoViewLib.dylib`App::prepareOptiX() + 366
        frame #20: 0x000000010000b8f1 OTracerTest`main(argc=3, argv=0x00007fff5fbfe4a8) + 321 at OTracerTest.cc:23
        frame #21: 0x00007fff89e755fd libdyld.dylib`start + 1
        frame #22: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 



 

Tips for Geometry Debug
--------------------------

* using propagation functions like ggv-reflect just adding a "--tracer" 
  option provides much faster OptiX renders by avoiding propagation overheads

* look for artifacts in OptiX render, especially from axial directions,
  to put viewpoint precisely in axes use eg: udp.py --eyew 0,500,0

* use discaxial torch with time fixed prior to intersections and color view 
  set to m2, can see immediately the directions from which rays are failing 
  to intersect

* review how code handles axial direction infinities, infinities arising from 
  attempted intersections against other planes can mess up intersection again
  

Wireframe view
----------------

* Interactor B key does m_scene->nextGeometryStyle();  BBOX/NORM/WIRE


Alternate Approaches
-----------------------

* https://github.com/nvpro-pipeline/pipeline
* http://on-demand.gputechconf.com/gtc/2015/presentation/S5148-Markus-Tavenrath.pdf


surface property debug, PMT id 
-------------------------------


ISSUE : seqmat first material mismatch to genstep material
-------------------------------------------------------------

Do not observe any seq trucation to 0xF mismatch, although that is possible.

npy- genstep_sequence_material_mismatch.py::

    In [156]: off = np.arange(len(s_first))[ s_first != p_gsgmat ]

    In [157]: off
    Out[157]: array([  3006,   8521,   8524, ..., 612838, 612839, 612840])

    In [158]: off.shape
    Out[158]: (104400,)

    In [159]: s_first.shape
    Out[159]: (612841,)


Many due to MI, but large chunk of gs:Gs sq:Ac 

TODO: make specialized indices categorizing these discrepancies to allow visualization  


ISSUE : genstep material index in wrong lingo 
----------------------------------------------

Genstep material index read into cs.MaterialIndex and used in wavelength lookups
as a standard line number cu/cerenkovstep.h::

    225         float4 props = wavelength_lookup(wavelength, cs.MaterialIndex);


G4StepNPY::applyLookup does a to b mapping between lingo which is invoked 
immediately after loading the genstep from file in App::loadGenstep::

     549     G4StepNPY genstep(npy);
     550     genstep.setLookup(m_loader->getMaterialLookup());
     551     if(!juno)
     552     {
     553         genstep.applyLookup(0, 2);   // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 
     554     }   


But dumping it, clear that not a line number::

    gs = stc_(1)

    In [33]: gsmat = gs.view(np.int32)[:,0,2]

    In [34]: gsmat
    Out[34]: array([12, 12, 12, ...,  8,  8,  8], dtype=int32)

    In [35]: np.unique(gsmat)
    Out[35]: array([ 1,  8, 10, 12, 13, 14, 19], dtype=int32)
           

Ahha the in file numbers need the lookup applied. 

Is the lookup accomodating the material code customization anyhow ?






RESOLVED photon level material code debug
------------------------------------------

Seeing some crazy materials, seems some parts of npy- not 
updated for the optical buffer approach::

   npy-dump 0 

* trace the material codes into OptiX from GBoundaryLib::createWavelengthAndOpticalBuffers
* simplify access to avoid such divergences

m_materials is GItemIndex which wraps Index::

     845             else if(psrc->isMaterial())
     846             {
     847                 m_materials->add(shortname.c_str(), psrc->getIndex() );  // registering source indices (aiScene mat index) into GItemIndex
     848                 unsigned int index_local = m_materials->getIndexLocal(shortname.c_str());
     849 
     850                 optical_data[opticalOffset + p*4 + optical_index]  = index_local ;
     851                 optical_data[opticalOffset + p*4 + optical_type]   =  0 ;
     852                 optical_data[opticalOffset + p*4 + optical_finish] =  0 ;
     853                 optical_data[opticalOffset + p*4 + optical_value]  =  0 ;



enhancement : COMPUTE mode
----------------------------

Next:

* pull out ok- App into separate file and make reusable 
  from tests/computeTest.cc with minimal duplication with main.cc
  
  * ie need to partition up compute from interop 

  * try to arrange use of same code no matter whether
    OpenGL or OptiX backed buffers are in use, need 
    some kind of facade to make these look the same 
    from the point of view of Thrust interop, CBufSpec 
    will help here

  * splitting the monolithic OEngine  


enhancement : interop with stream compaction, deferred host allocation 
-----------------------------------------------------------------------

Next:

* revive host based detailed debug dumping with PhotonsNPY RecordNPY 

* look into photon picking ? maybe provide a GUI to invoke the pullbacks (perhaps partial)
  and run the detailed debug

* test operation without any host allocations

* implement stream compaction "partial" pullbacks based on internal criteria or external mask  
  following testcode of optixthrust-



RESOLVED issue: jpmt timeouts binary search to pin down 
---------------------------------------------------------

* problem was a genstep with parameters causing an infinite loop
  in cerenkov generation wavelength sampling 

  * solution was protections to prevent sin2Theta going negative



::

    ggv --juno 
       # no pmt, evt propagation vis not working 

    ggv --jpmt --modulo 1000 
       # causes a timeout+freeze requiring a reboot

    ggv --jpmt --modulo 1000 --nopropagate
       # can visualize jpmt OptiX geometry: and it looks OK

    ggv --jpmt --modulo 1000 --trivial
       # swap generate program with a trivial standin  : works 

    ggv --jpmt --modulo 1000 --bouncemax 0
       # just generate, no propagation : timeout+freeze, reboot
       
    ggv --jpmt --modulo 1000 --trivial
       # progressively adding lines from generate into trivial
       # suggests first issue inside generate_cerenkov_photon/wavelength_lookup

    ggv --jpmt --modulo 1000 --bouncemax 0
       # just generate, no propagation : works after kludging wavelength_lookup 
       # to always give a constant valid float4

    ggv --jpmt --modulo 100
       #  still with kludged wavelength_lookup : works, with photon animation operational
       #
       # interestingly performance of OptiX and OpenGL geometry visualizations
       # are about the same with the full JUNO geometry 2-3 fps, 
       # with DYB OpenGL vis is much faster than OptiX appoaching: 60 fps  

    ggv --jpmt --modulo 50
       #  still with kludged wavelength_lookup : timeout...  maybe stepping off reservation somewhere else reemission texture ?

    ggv --jpmt --modulo 100 --override 1 
       # putting back the wavelength_look get timeout even when override to a single photon

    ggv --jpmt --modulo 100 --override 1 --trivial
       # with trivial prog doing wavelength dumping

    ggv --make --jpmt --modulo 100 --override 1
       # with bounds checking on wavelength lookup succeed with single photon, but not without the override

    ggv --make --jpmt --modulo 100 --override 5181
       # with bounds checking on wavelength lookup succeed with override 5181, failing at override 5182
       #    photon_id = 5181  is doing something naughty

    ggv --make --jpmt --modulo 100 --override 5181 --debugidx 5180
       # check on the photon before, which works
       #
       #
       # [2015-Aug-31 16:23:50.594320]: OEngine::generate OVERRIDE photon count for debugging to 5181
       #  generate debug photon_id 5180 genstep_id 18 ghead.i.x -18001 
       #  cs.Id -18001 ParentId 1 MaterialIndex 48 NumPhotons 39 
       #

    ggv --make --jpmt --modulo 100 --override 5182 --debugidx 5181 --bouncemax 0 
       # now the one that fails, with propagation inhibited  : still failing 

    ggv --make --jpmt --modulo 100 --override 5182 --debugidx 5181 
       # try with kludge skipping of Aluminium : works, so can dump nemesis 
       #
       # ... hmm refractive index of 1.000 for a metal 
       #
       # [2015-Aug-31 16:45:04.481506]: OEngine::generate count 0 size(10406,1)
       # [2015-Aug-31 16:45:04.481600]: OEngine::generate OVERRIDE photon count for debugging to 5182
       # generate debug photon_id 5181 genstep_id 19 ghead.i.x -19001 
       # cs.Id -19001 ParentId 1 MaterialIndex 24 NumPhotons 282 
       # x0 -15718.109375 -2846.020996 -9665.920898  t0 62.278240 
       # DeltaPosition -1.087246 -0.197473 -0.667886  step_length 1.291190  
       # code 13  charge -1.000000 weight 1.000000 MeanVelocity 299.792267 
       # BetaInverse 1.000001  Pmin 0.000002 Pmax 0.000015 maxCos 0.751880 
       # maxSin2 0.434676  MeanNumberOfPhotons1 232.343796 MeanNumberOfPhotons2 232.343796 MeanNumberOfPhotonsMax 232.343796 
       # p0 -0.842050 -0.152938 -0.517264  
       # cscheck sample wavelength lo/mi/hi   59.999996 111.724136 810.000122 
       # cscheck sample rindex lo/mi/hi   1.000000 1.000000 1.000000 
       # cscheck sample abslen lo/mi/hi   1000000.000000 1000000.000000 1000000.000000 
       # cscheck sample scalen lo/mi/hi   1000000.000000 1000000.000000 1000000.000000 
       # cscheck sample reempr lo/mi/hi   0.000000 0.000000 0.000000 
       #

    ggv --make --jpmt --modulo 100 
       #
       # fix by modifying cerernkovstep.h wavelength sampling loop 
       # to avoid sin2Theta from going -ve 
       #
       #      sin2Theta = fmaxf( 0.0001f, (1.f - cosTheta)*(1.f + cosTheta));  
       #
       # TODO: check for artifacts in wavelength distribution
 
   ggv --make --jpmt
       #
       # at modulo 10,    propagate time is 1.009s  
       # at modulo  5,    propagate time is 1.740s
       # at full genstep, propagate time is 7.053s 



Do things go bad on a genstep boundary ?

::

    In [1]: c = np.load("/usr/local/env/juno/cerenkov/1.npy")

    In [7]: c.view(np.int32)[:,0,3][::100].sum()    # number of photons, modulo scaled down and summed matches log 
    Out[7]: 10406

::
    In [8]: cc = c.view(np.int32)[:,0,3][::100].cumsum()    # genstep index 18 has cumsum 5181
    Out[8]: 
    array([  322,   607,   883,  1164,  1476,  1513,  1831,  2160,  2462,
            2776,  3078,  3375,  3699,  4002,  4310,  4603,  4881,  5142,
            5181,  5463,  5776,  6052,  6346,  6628,  6636,  6646,  6942,
            7235,  7521,  7817,  8123,  8399,  8695,  9012,  9295,  9584,
            9777, 10068, 10406])


    In [34]: c[::100][:,0].view(np.int32)
    Out[34]: 
    array([[    -1,      1,     48,    322],    Id/ParentId/MaterialIndex/NumPhotons
           [ -1001,      1,     48,    285],
           [ -2001,      1,     48,    276],
           [ -3001,      1,     48,    281],
           ...


    In [50]: for i,_ in enumerate(c[::100][:,0].view(np.int32)):print i,_,cc[i]
    0  [ -1         1     48    322] 322
    1  [-1001       1     48    285] 607
    2  [-2001       1     48    276] 883
    3  [-3001       1     48    281] 1164
    4  [-4001       1     48    312] 1476     #  48:Water, 24:Aluminium, 42:Tyvek 
    5  [-5001       1     48     37] 1513
    6  [-6001       1     48    318] 1831
    7  [-7001       1     48    329] 2160
    8  [-8001       1     48    302] 2462
    9  [-9001       1     48    314] 2776
    10 [-10001      1     48    302] 3078
    11 [-11001      1     48    297] 3375
    12 [-12001      1     48    324] 3699
    13 [-13001      1     48    303] 4002
    14 [-14001      1     48    308] 4310
    15 [-15001      1     48    293] 4603
    16 [-16001      1     48    278] 4881
    17 [-17001      1     48    261] 5142
    18 [-18001      1     48     39] 5181    ### genstep index 18 ends with photon_id 5180
    19 [-19001      1     24    282] 5463
    20 [-20001      1     24    313] 5776
    21 [-21001      1     24    276] 6052
    22 [-22001      1     24    294] 6346
    23 [-23001      1     24    282] 6628
    24 [-24001      1     24      8] 6636
    25 [-25001   4720     24     10] 6646
    26 [-26001   1553     48    296] 6942
    27 [-27001   4964     48    293] 7235
    28 [-28001   5540     42    286] 7521
    29 [-29001   1552     48    296] 7817
    30 [-30001   6048     48    306] 8123
    31 [-31001   6464     48    276] 8399
    32 [-32001   1156     48    296] 8695
    33 [-33001   1050     48    317] 9012
    34 [-34001   6977     48    283] 9295
    35 [-35001    692     48    289] 9584
    36 [-36001    456     48    193] 9777
    37 [-37001    222     48    291] 10068
    38 [-38001    106     48    338] 10406



issue: jpmt wavelengthBuffer/boundarylib ? maybe bad material indices ?
-------------------------------------------------------------------------

* is the cs.MaterialIndex expected to be the wavelength texture line number ?

  * if so then the jpmt/juno numbers do need a "translation" applied ?
  * GBoundaryLibMetadata.json has 18 boundaries 0..17

::

    In [5]: cd /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae
    /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae

    In [6]: a = np.load("wavelength.npy")

    In [40]: a.reshape(-1,6,39,4).shape
    Out[40]: (18, 6, 39, 4)

    In [47]: a.reshape(-1,6,39,4)[6]
    Out[47]: 
    array([[[       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.345,      273.208,  1000000.   ,        0.   ],
            [       1.36 ,      273.208,  1000000.   ,        0.   ],
            [       1.375,      273.208,  1000000.   ,        0.   ],
            [       1.39 ,      691.558,  1000000.   ,        0.   ],
            [       1.384,     1507.119,  1000000.   ,        0.   ],

    In [54]: 18*6
    Out[54]: 108

::

    delta:npy blyth$ /usr/local/env/numerics/npy/bin/NPYTest
    [2015-08-28 21:14:53.244421] [0x000007fff7650e31] [debug]   NPY<T>::load /usr/local/env/juno/cerenkov/1.npy
    G4StepNPY
     ni 3840 nj 6 nk 4 nj*nk 24 
     (    0,    0)               -1                1               48              322  sid/parentId/materialIndex/numPhotons 
     (    0,    1)            0.000            0.000            0.000            0.000  position/time 
     (    0,    2)           -0.861           -0.156           -0.530            1.023  deltaPosition/stepLength 
     (    0,    3)               13           -1.000            1.000          299.792  code 
     (    0,    4)            1.000            0.000            0.000            0.688 
     (    0,    5)            0.527          293.245          293.245            0.000 
     ( 3839,    0)           -38391                4               48               47  sid/parentId/materialIndex/numPhotons 
     ( 3839,    1)          -16.246           -2.947          -10.006            0.064  position/time 
     ( 3839,    2)           -0.191           -0.194            0.236            0.378  deltaPosition/stepLength 
     ( 3839,    3)               11           -1.000            1.000          230.542  code 
     ( 3839,    4)            1.300            0.000            0.000            0.895 
     ( 3839,    5)            0.200          165.673          110.064            0.000 
     24 
     42 
     48 
     24 : 750 
     42 : 52 
     48 : 3038 




Initial values of material indices are not unreasonable, maybe problem on subsequent steps::

    simon:ggeoview blyth$ ok-detector-jpmt
    simon:ggeo blyth$ ggeo-blt 24 42 48
    /usr/local/env/optix/ggeo/bin/GBoundaryLibTest 24 42 48
    GCache::readEnvironment setting IDPATH internally to /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae 
    [2015-08-31 12:54:10.546692] [0x000007fff77ea531] [warning] GBoundaryLib::setWavelengthBuffer didnt see 54, numBoundary: 18
    GBoundaryLib::loadBoundary digest mismatch 7 : d1a3424507d661c74ab51c4b5c82dff0 202bc56442e88df7f4be6f3af62acf70 
    GBoundaryLib::loadBoundary digest mismatch 13 : 8dc0d036da7ed8b5d4606cfe506a82f7 82a76e8ae56ac49dc00174734af2d8b8 
    GBoundaryLib::loadBoundary digest mismatch 14 : ac621cac48edd9555db9b8f9f5f56015 1bb254d022a246eb98cef4846123154e 
    GBoundaryLib::loadBoundary digest mismatch 15 : c3baf1e9325fac7e81b218e23804557d 39b93748d45456bc1aa6cb0e326f0fd3 
    boundary : index  0 dede45b90304e0f9dd9c7c5edce7c8b1 Galactic/Galactic/-/- 
    boundary : index  1 124d278374f95ec3742e1268e6e8f478 Rock/Galactic/-/- 
    boundary : index  2 4befaffca91e8cb0fd5662ae2d81bd65 Air/Rock/-/- 
    boundary : index  3 231c44f02f80c88638cb09dff25df5f6 Air/Air/-/- 
    boundary : index  4 576a076a3f1f332dad075d3c2d8181d7 Aluminium/Air/-/- 
    boundary : index  5 eb855bbd039a6401bfacc6202ea5034c Steel/Rock/-/- 
    boundary : index  6 d18726a8d2660e6be4b8ae326bd38ee6 Water/Steel/-/- 
    boundary : index  7 d1a3424507d661c74ab51c4b5c82dff0 Tyvek/Water/-/CDTyvekSurface 
    boundary : index  8 608795d154c5752988d6882d87de18e6 Water/Tyvek/-/- 
    boundary : index  9 1118e140d2fe2dc9f07c350302e5ee1e Acrylic/Water/-/- 
    boundary : index 10 9ad9179c5dc8584ab0a68f460dbfddde LS/Acrylic/-/- 
    boundary : index 11 da505cbe2bdfaa95b091f31761d81a93 Pyrex/Water/-/- 
    boundary : index 12 11467e52d1bc229355bf173f871790d2 Pyrex/Pyrex/-/- 
    boundary : index 13 8dc0d036da7ed8b5d4606cfe506a82f7 Vacuum/Pyrex/-/PMT_20inch_photocathode_logsurf2 
    boundary : index 14 ac621cac48edd9555db9b8f9f5f56015 Vacuum/Pyrex/PMT_20inch_mirror_logsurf1/- 
    boundary : index 15 c3baf1e9325fac7e81b218e23804557d Vacuum/Pyrex/-/PMT_3inch_photocathode_logsurf2 
    boundary : index 16 00fbb4643f7986d8c5f1499d5b3b3e22 Steel/Water/-/- 
    boundary : index 17 47a41d6b6a602cc04be06523254ec39c Copper/Water/-/- 

    GBoundaryLib.dumpWavelengthBuffer 24 
    GBoundaryLib::dumpWavelengthBuffer wline 24 numSub 18 domainLength 39 numQuad 6 

      24 |   4/  0 Aluminium0x22ca560 
               1.000           1.000           1.000           1.000           1.000           1.000           1.000           1.000
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000

    GBoundaryLib.dumpWavelengthBuffer 42 
    GBoundaryLib::dumpWavelengthBuffer wline 42 numSub 18 domainLength 39 numQuad 6 

      42 |   7/  0 Tyvek0x229f920 
               1.000           1.000           1.000           1.000           1.000           1.000           1.000           1.000
           10000.000       10000.000       10000.000       10000.000       10000.000       10000.000       10000.000       10000.000
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000

    GBoundaryLib.dumpWavelengthBuffer 48 
    GBoundaryLib::dumpWavelengthBuffer wline 48 numSub 18 domainLength 39 numQuad 6 

      48 |   8/  0 Water0x22c0a30 
               1.330           1.360           1.372           1.357           1.352           1.346           1.341           1.335
             273.208         273.208        3164.640       12811.072       28732.207       13644.791        2404.398         371.974
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    simon:ggeo blyth$ 




issue: ~/jpmt_mm0_too_many_vertices.txt
------------------------------------------

1.79M vertices for jpmt mm0 (global) seems excessive, either missing a repeater or some bug.::

    ggv -G --jpmt

    120 [2015-Aug-25 18:52:37.665158]: GMergedMesh::create index 0 from default root base lWorld0x22ccd90
    121 [2015-Aug-25 18:52:37.730168]: GMergedMesh::create index 0 numVertices 1796042 numFaces 986316 numSolids 289733 numSolidsSelected 1032


From m_mesh_usage in GGeo and GMergedMesh sStrut and sFasteners are the culprits::

    [2015-Aug-25 19:50:40.251333]: AssimpGGeo::convertMeshes  i   19 v  312 f  192 n sStrut0x304f210
    [2015-Aug-25 19:50:40.251575]: AssimpGGeo::convertMeshes  i   20 v 3416 f 1856 n sFasteners0x3074ea0

    [2015-Aug-25 19:54:01.663594]: GMergedMesh::create index 0 from default root base lWorld0x22ccd90
    [2015-Aug-25 19:54:07.339150]: GMergedMesh::create index 0 numVertices 1796042 numFaces 986316 numSolids 289733 numSolidsSelected 1032
    GLoader::load reportMeshUsage (global)
         5 :     62 : sWall0x309ce60 
         6 :      1 : sAirTT0x309cbb0 
         7 :      1 : sExpHall0x22cdb00 
         8 :      1 : sTopRock0x22cd500 
         9 :      1 : sTarget0x22cfbd0 
        10 :      1 : sAcrylic0x22cf9a0 
        19 :    480 : sStrut0x304f210 
        20 :    480 : sFasteners0x3074ea0 
        21 :      1 : sInnerWater0x22cf770 
        22 :      1 : sReflectorInCD0x22cf540 
        23 :      1 : sOuterWaterPool0x22cef90 
        24 :      1 : sSteelTub0x22ce610 
        25 :      1 : sBottomRock0x22cde40 
            ---------

    In [7]: 480+480+62+10
    Out[7]: 1032          ## matches numSolidsSelected

    In [5]: 3416*480+312*480
    Out[5]: 1789440


::

    simon:juno blyth$ grep sFasteners t3.dae
        <geometry id="sFasteners0x3074ea0" name="sFasteners0x3074ea0">
            <source id="sFasteners0x3074ea0-Pos">
              <float_array count="2742" id="sFasteners0x3074ea0-Pos-array">
                <accessor count="914" source="#sFasteners0x3074ea0-Pos-array" stride="3">
            <source id="sFasteners0x3074ea0-Norm">
              <float_array count="5184" id="sFasteners0x3074ea0-Norm-array">
                <accessor count="1728" source="#sFasteners0x3074ea0-Norm-array" stride="3">
            <source id="sFasteners0x3074ea0-Tex">
              <float_array count="2" id="sFasteners0x3074ea0-Tex-array">
                <accessor count="1" source="#sFasteners0x3074ea0-Tex-array" stride="2">
            <vertices id="sFasteners0x3074ea0-Vtx">
              <input semantic="POSITION" source="#sFasteners0x3074ea0-Pos"/>
              <input offset="0" semantic="VERTEX" source="#sFasteners0x3074ea0-Vtx"/>
              <input offset="1" semantic="NORMAL" source="#sFasteners0x3074ea0-Norm"/>
              <input offset="2" semantic="TEXCOORD" source="#sFasteners0x3074ea0-Tex"/>
              <meta id="sFasteners0x3074ea0">
          <instance_geometry url="#sFasteners0x3074ea0">
    simon:juno blyth$ 



Contiguous block of Fasteners all leaves at depth 6::

    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | wc -l
         480
    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | head -1
       282429 [  6:54799/55279]    0          lFasteners0x3075090   
    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | tail -1
       282908 [  6:55278/55279]    0          lFasteners0x3075090   

    In [9]: 282429+480-1
    Out[9]: 282908


TODO: 


* on dyb GTreeCheck::findRepeatCandidates 

  * not restricting repeats to non-leaf looses some geometry
  * but putting it back gives PMTs in both instance0 and instance1  
  * GTreeCheck.dumpTree ridx not making sense when allow leaf repeats 

* dump the text node tree for juno, to see where sFasteners is 
* add --repeatidx 0,1,2,3 controlled loading in GGeo::loadMergedMeshes etc..
  so can skip the problematic extremely large 0




squeeze approaches for jpmt
----------------------------

* remove vertex color, do at solid/boundary level
* compress vertex normals 
* reuse vertex structures for OptiX ?



computeTest with different core counts controlled via CUDA_VISIBLE_DEVICES
----------------------------------------------------------------------------

Juno Scintillation 2, genstep scaledown 25
--------------------------------------------

::

    genstepAsLoaded : 4e16b039dc40737a4c0c51d7b213a118
    genstepAfterLookup : 4e16b039dc40737a4c0c51d7b213a118
               Type :   scintillation
                Tag :               1
           Detector :            juno
        NumGensteps :            1774
             RngMax :         3000000
         NumPhotons :         1493444
         NumRecords :        14934440
          BounceMax :               9
          RecordMax :              10
        RepeatIndex :              10
         photonData : 33b5c1f991b46e09036e38c110e36102
         recordData : 55a15aacf09d4e8dcf269d6e882b481e
       sequenceData : 035310267fc2a678f2c8cad2031d7101




::

    2.516              GT 750M          ggv.sh --cmp --juno -s 
 
    0.487              GTX 750 Ti 


    0.153      -         Tesla K40m  ( 11520 )

    0.157      0,1,2,3 

                      
              Tesla K40m   (5760)

    0.201      0,1                      
    0.200      2,3                      

    0.179      1,2                     
    0.179      0,2                      
    0.178      1,3                      
     
    0.202      0,1,2
    0.201      0,1,3

    0.134      1,2,3
 


::

    In [1]: 2.516/0.134
    Out[1]: 18.776119402985074




Juno Cerenkov 1, scaledown ?10
---------------------------------

::

    0.126,0.126   0         Tesla K40m  2880 CUDA cores  
    0.127         1
    0.127         2
    0.126         3
  
    0.088,0.087   0,1             5760 
    0.076         0,2
    0.099         2,3
    0.080         1,3

    0.076         0,1,2           8640
    0.058         1,2,3
    0.057         1,2,3

    0.062         0,1,2,3         11520
    0.062,0.062,0.062,0.063   NO ENVVAR
    

    1.130          GT750M    ggv.sh --juno --cmp      384 CUDA cores
    1.143 
    1.146 
    1.137 
    1.139 


    0.195,0.197    GTX 750 Ti    640 CUDA Cores                             


    a = np.array( [[384, 1.130],[640,0.195],[2880,0.126],[5760,0.080],[8640,0.070],[11520,0.062]] )

    plt.plot( a[:,0], a[0,-1]/a[:,1], "*-")



GGeoview Compute 
------------------

Compute only mode::

   ok-compute -b0
   ok-compute -b1
   ok-compute -b2
   ok-compute -b4   # up to 4 bounces working 

   ok-compute -b5   # crash for beyond 4  


Usage tips
-----------

Thoughts on touch mode : OptiX single-ray-cast OR OpenGL depth buffer/unproject 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OptiX based touch mode, is not so useful operationally (although handy as a debug tool) as:

#. it requires to do an OptiX render before it can operate
#. will usually be using OpenGL rendering to see the geometry often with 
   clipping planes etc.. that only OpenGL knows about.  

Thus need an OpenGL depth buffer unproject approach too.


Low GPU memory running
~~~~~~~~~~~~~~~~~~~~~~~~~~

When GPU memory is low OptiX startup causes a crash, 
to run anyhow disable OptiX with::

    ok-run --optixmode -1

To free up GPU memory restart the machine, or try sleep/unsleep and
exit applications including Safari, Mail that all use GPU memory. 
Observe that sleeping for ~1min rather than my typical few seconds 
frees almost all GPU memory.

Check available GPU memory with **cu** if less than ~512MB OptiX will
crash at startup::

    delta:optixrap blyth$ t cu
    cu is aliased to cuda_info.sh


Clipping Planes and recording frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    udp.py --cutnormal 1,0,0 --eye -2,0,0 --cutpoint 0,0,0
    udp.py --cutnormal 1,1,0 --cutpoint -0.1,-0.1,0


Although model frame coordinates are useful for 
intuitive data entry the fact that the meaning is relative
to the selected geometry makes them less 
useful as a way of recording a plane, 
so record planes in world frame coordinates.

This would allow to find the precise plane that 
halves a piece of geometry by selecting that
and providing a way to accept world planes, could 
use --cutplane x,y,x,w to skip the model_to_world 
conversion.

The same thinking applies to recording viewpoint bookmarks.


New way of interpolated photon position animation ?
----------------------------------------------------

See oglrap- for untested idea using geometry shaders alone.


Old way of doing interpolated photon position animation
-----------------------------------------------------------

* splayed out by maxsteps VBO

* recorded each photon step into its slot 

* pre-render CUDA time-presenter to find before and after 
  positions and interpolate between them writing into specific top slot 
  of the splay.


Problems:

* limited numbers of particles can be animated (perhaps 1000 or so)
  as approach multiplies storage by the max number of steps are kept

* most of the storage is empty, for photons without that many steps 

Advantages:

* splaying out allows CUDA to operate fully concurrently 
  with no atomics complexities, as every photon step has its place 
  in the structure 

* OpenGL can address and draw the VBO using fixed offsets/strides
  pointing at the interpolated slot, geometry shaders can be used to 
  amplify a point and momentum direction into a line


Package Dependencies Tree of GGeoView
--------------------------------------

* higher level repeated dependencies elided for clarity 

::

    NPY*   (~11 classes)
       Boost
       GLM         

    Cfg*  (~1 class)
       Boost 

    numpyserver*  (~7 classes)
       Boost.Asio
       ZMQ
       AsioZMQ
       Cfg*
       NPY*

    cudawrap* (~5 classes)
       CUDA

    Opticks
       Cfg 
 
    GGeo*  (~22 classes)
       NPY*

    AssimpWrap* (~7 classes)
       Assimp
       GGeo* 

    OGLRap*  (~29 classes)
       GLEW
       GLFW
       ImGui
       AssimpWrap*
       Cfg*
       NPY*

    OptiXRap* (~7 classes)
       OptiX
       OGLRap*
       AssimpWrap*
       GGeo*    
   


Data Flow thru the app
-------------------------

* Gensteps NPY loaded from file (or network)

* main.NumpyEvt::setGenstepData 

  * determines num_photons
  * allocates NPY arrays for photons, records, sequence, recsel, phosel
    and characterizes content with MultiViewNPY 

* main.Scene::uploadEvt

  * gets genstep, photon and record renderers to upload their respective buffers 
    and translate MultiViewNPY into OpenGL vertex attributes

* main.Scene::uploadSelection

  * recsel upload
  * hmm currently doing this while recsel still all zeroes 

* main.OptiXEngine::initGenerate(NumpyEvt* evt)

  * populates OptiX context, using OpenGL buffer ids lodged in the NPY  
    to create OptiX buffers for each eg::

        m_genstep_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, genstep_buffer_id);

* main.OptiXEngine::generate, cu/generate.cu

  * fills OptiX buffers: photon_buffer, record_buffer, sequence_buffer

* main.Rdr::download(NPY*)

  * pullback to host NPY the VBO/OptiX buffers using Rdr::mapbuffer 
    Rdr::unmapbuffer to get void* pointers from OpenGL

    * photon, record and sequence buffers are downloaded

* main.ThrustArray::ThrustArray created for: sequence, recsel and phosel 

  * OptiXUtil::getDevicePtr devptr used to allow Thrust to access these OpenGL buffers 
    
* main.ThrustIdx indexes the sequence outputing into phosel and recsel

  * recsel is created from phosel using ThrustArray::repeat_to

* main.Scene::render Rdr::render for genstep, photon, record 

  * glBindVertexArray(m_vao) and glDrawArrays 
  * each renderer has a single m_vao which contains buffer_id and vertex attributes


Issue: recsel changes not seen by OpenGL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The zeroed recsel buffer was uploaded early, it was modified
with Thrust using the below long pipeline but the 
changes to the device buffer where not seen by OpenGL

* NumpyEvt create NPY
* Scene::uploadEvt, Scene::uploadSelection - Rdr::upload (setting buffer_id in the NPY)
* OptiXEngine::init (convert to OptiX buffers)
* OptiXUtil provides raw devptr for use by ThrustArray
* Rdr::render draw shaders do not see the changes to the recsel buffer 

Workaround by simplifying pipeline for non-conforming buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recsel, phosel do not conform to the pattern of other buffers 

* not needed by OptiX
* only needed in host NPY for debugging 
* phosel is populated on device by ThrustIdx::makeHistogram from the OptiX filled sequence buffer
* recsel is populated on device by ThrustArray::repeat_to on phosel 

Formerly had no way to get buffers into Thrust other than 
going through the full pipeline. Added capability to ThrustArray 
to allocate/resize buffers allowing simpler flow:

* NumpyEvt create NPY (recsel, phosel still created early on host, but they just serve as dimension placeholders)
* allocate recsel and phosel on device with ThrustArray(NULL, NPY dimensions), populate with ThrustIdx
* ThrustArray::download into the recsel and phosel NPY 
* Scene::uploadSelection to upload with OpenGL for use from shaders 

TODO: skip redundant Thrust download, OpenGL upload using CUDA/OpenGL interop ?



C++ library versions
----------------------

::

    delta:~ blyth$ otool -L $(ok-;ok-deps) | grep c++
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 60.0.0)



Pre-cook RNG Cache
-------------------

* currently the work number must precicely match the hardcoded 
  value used for OptiXEngine::setRngMax  

  * TODO: tie these together via envvar


::

    delta:ggeoview blyth$ ok-rng-prep
    cuRANDWrapper::instanciate with cache enabled : cachedir /usr/local/env/ggeoview.build/lib/rng
    cuRANDWrapper::Allocate
    cuRANDWrapper::InitFromCacheIfPossible
    cuRANDWrapper::InitFromCacheIfPossible : no cache initing and saving 
    cuRANDWrapper::Init
     init_rng_wrapper sequence_index   0  thread_offset       0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   138.0750 ms 
    ...



Improvement
-------------

Adopt separate minimal VBO for animation 

* single vec4 (position, time) ? 
* no need for direction as see that from the interpolation, 
* polz, wavelength, ... keep these in separate full-VBO for detailed debug 
  of small numbers of stepped photons 


Does modern OpenGL have any features that allow a better way
--------------------------------------------------------------

* http://gamedev.stackexchange.com/questions/20983/how-is-animation-handled-in-non-immediate-opengl

  * vertex blend-based animation
  * vertex blending
  * use glVertexAttribPointer to pick keyframes, 
  * shader gets two "position" attributes, 
    one for the keyframe in front of the current 
    time and one for the keyframe after and a uniform that specifies 
    how much of a blend to do between them. 

hmm not so easy for photon simulation as they all are on their own timeline :
so not like traditional animation keyframes
 


glDrawArraysIndirect introduced in 4.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.opengl.org/wiki/History_of_OpenGL#OpenGL_4.1_.282010.29
* https://www.opengl.org/wiki/Vertex_Rendering#Indirect_rendering
* http://stackoverflow.com/questions/5047286/opengl-4-0-gpu-draw-feature
* https://www.opengl.org/registry/specs/ARB/draw_indirect.txt

Indirect rendering is the process of issuing a drawing command to OpenGL,
except that most of the parameters to that command come from GPU storage
provided by a Buffer Object.
The idea is to avoid the GPU->CPU->GPU round-trip; the GPU decides what range
of vertices to render with. All the CPU does is decide when to issue the
drawing command, as well as which Primitive is used with that command.

The indirect rendering functions take their data from the buffer currently
bound to the GL_DRAW_INDIRECT_BUFFER binding. Thus, any of these
functions will fail if no buffer is bound to that binding.

So can tee up a buffer of commands GPU side, following layout::

    void glDrawArraysIndirect(GLenum mode, const void *indirect);

    typedef  struct {
       GLuint  count;
       GLuint  instanceCount;
       GLuint  first;
       GLuint  baseInstance;   // MUST BE 0 IN 4.1
    } DrawArraysIndirectCommand;

Where each cmd is equivalent to::

    glDrawArraysInstancedBaseInstance(mode, cmd->first, cmd->count, cmd->instanceCount, cmd->baseInstance);

Similarly for indirect indexed drawing::

    glDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect);

    typedef  struct {
        GLuint  count;
        GLuint  instanceCount;
        GLuint  firstIndex;
        GLuint  baseVertex;
        GLuint  baseInstance;
    } DrawElementsIndirectCommand;

With each cmd equivalent to:: 

    glDrawElementsInstancedBaseVertexBaseInstance(mode, cmd->count, type,
      cmd->firstIndex * size-of-type, cmd->instanceCount, cmd->baseVertex, cmd->baseInstance);

* https://www.opengl.org/sdk/docs/man/html/glDrawElementsInstancedBaseVertex.xhtml


EOU
}

ok-env(){     
    olocal- 
    #optix-
    #optix-export
    #opticks-
}


ok-sdir(){ echo $(opticks-home)/ok; }
ok-tdir(){ echo $(opticks-home)/ok/tests ; }
ok-idir(){ echo $(opticks-idir) ; }
ok-bdir(){ echo $(opticks-bdir)/ok; }

ok-c(){    cd $(ok-sdir)/$1 ; }
ok-cd(){   cd $(ok-sdir)/$1 ; }
ok-scd(){  cd $(ok-sdir); }
ok-tcd(){  cd $(ok-tdir); }
ok-icd(){  cd $(ok-idir); }
ok-bcd(){  cd $(ok-bdir); }


ok-gdir(){ echo $(ok-idir).generated ; }
ok-bindir(){  echo $(ok-idir)/bin ; }
ok-bin(){ echo ${OPTICKS_BINARY:-$(ok-idir)/bin/$(ok-name)} ; }

#ok-rng-dir(){ echo $(ok-bdir)/lib/rng ; }  gets deleted too often for keeping RNG 
ok-rng-dir(){ echo $(ok-idir)/cache/rng ; }

ok-ptx-dir(){ echo $(ok-bdir)/lib/ptx ; }
ok-rng-ls(){  ls -l $(ok-rng-dir) ; }
ok-ptx-ls(){  ls -l $(ok-ptx-dir) ; }

ok-name(){ echo OK ; }
ok-tag(){  echo OK ; }

ok-apihh(){  echo $(ok-sdir)/$(ok-tag)_API_EXPORT.hh ; }
ok---(){     touch $(ok-apihh) ; ok--  ; }


ok-edit(){     vi $(ok-sdir)/ggv.sh ; }

ok-wipe(){
   local bdir=$(ok-bdir)
   rm -rf $bdir
}


ok--(){        opticks--     $(ok-bdir) ; }
ok-t(){        opticks-t   $(ok-bdir) $* ; }
ok-ti(){       opticks-ti  $(ok-bdir) $* ; }
ok-genproj(){  ok-scd ; opticks-genproj $(ok-name) $(ok-tag) ; }
ok-gentest(){  ok-tcd ; opticks-gentest ${1:-Scene} $(ok-tag) ; }
ok-txt(){      vi $(ok-sdir)/CMakeLists.txt $(ok-tdir)/CMakeLists.txt ; }




ok-accelcache()
{
    ok-export
    ls -l ${DAE_NAME_DYB/.dae}.*.accelcache
}
ok-accelcache-rm()
{
    ok-export
    rm ${DAE_NAME_DYB/.dae}.*.accelcache
}




ok-idpath()
{
   ok-
   ok-run --idpath 2>/dev/null 
}

ok-steal-bookmarks()
{
   local idpath=$(ok-idpath)
   cp ~/.g4daeview/dyb/bookmarks20141128-2053.cfg $idpath/bookmarks.ini
}



ok-export()
{
   local msg="=== $FUNCNAME :"

   export-
   export-export

   [ "$OPTICKS_GEOKEY" == "" ] && echo $msg MISSING ENVVAR OPTICKS_GEOKEY && sleep 10000000
   #[ "$OPTICKS_QUERY"  == "" ] && echo $msg MISSING ENVVAR OPTICKS_QUERY && sleep 10000000
   #[ "$OPTICKS_CTRL" == "" ]   && echo $msg MISSING ENVVAR OPTICKS_CTRL && sleep 10000000

   unset SHADER_DIR 
   unset SHADER_DYNAMIC_DIR 
   unset SHADER_INCL_PATH

   unset RAYTRACE_PTX_DIR
   unset RAYTRACE_RNG_DIR

   export CUDAWRAP_RNG_MAX=$(ok-rng-max)
} 

ok-export-dump()
{
   env | grep OPTICKS
   env | grep SHADER
   env | grep RAYTRACE
   env | grep CUDAWRAP

}

ok-run(){ 
   local bin=$(ok-bin)
   ok-export
   if [ "${bin: -3}" == ".py" ]; then 
      python $bin $* 
   else    
      $bin $*
   fi
}



ok-compute(){ 
   local bin=$(ok-compute-bin)
   ok-export
   $bin $*
}

ok-compute-lldb(){ 
   local bin=$(ok-compute-bin)
   ok-export
   lldb $bin $*
}

ok-compute-gdb(){ 
   local bin=$(ok-compute-bin)
   ok-export
   gdb --args $bin $*
}



ok-vrun(){ 
   local bin=$(ok-bin)
   ok-export
   vglrun $bin $*
}

ok-gdb(){ 
   local bin=$(ok-bin)
   ok-export
   gdb --args $bin $*
}

ok-valgrind(){ 
   local bin=$(ok-bin)
   ok-export
   valgrind $bin $*
}

ok-lldb()
{
   local bin=$(ok-bin)
   ok-export
   lldb $bin -- $*
}

ok-dbg()
{
   case $(uname) in
     Darwin) ok-lldb $* ;;
          *) ok-gdb  $* ;;
   esac
}



ok-dep-action()
{
    local action=${1:-install}
    ok-deps- | while read dep 
    do
        $dep-
        $dep-$action
    done
}

ok-dep-cmake(){   ok-dep-action cmake ; }
ok-dep-install(){ ok-dep-action install ; }
ok-dep-wipe(){    ok-dep-action wipe ; }


ok-full()
{

   ok-dep-wipe

   opticks-cmake
   opticks-install

   #ok-dep-install
}


ok-deps-(){ cat << EOD
bcfg
bregex
npy
optickscore
ggeo
assimpwrap
openmeshrap
ppm
oglrap
cudawrap
thrustrap
optixrap
opticksop
opticksgl
ok
EOD
}

ok-deps(){
   local suffix=${1:-dylib}
   local dep
   $FUNCNAME- | while read dep ; do
       $dep-
       #printf "%30s %30s \n" $dep $($dep-idir) 
       echo $($dep-idir)/lib/*.${suffix}
   done
}

ok-ls(){   ls -1 $(ok-;ok-deps) ; }
ok-libs(){ otool -L $(ok-;ok-deps) ; }

ok-linux-setup() {
    local dep
    local edeps="boost glew glfw imgui glm assimp"
    local deps="$edeps bcfg bregex npy ggeo assimpwrap ppm oglrap cudawrap optixrap thrustrap"
    for dep in $deps
    do
        $dep-
        [ -d "$($dep-idir)/lib" ] &&  export LD_LIBRARY_PATH=$($dep-idir)/lib:$LD_LIBRARY_PATH
        [ -d "$($dep-idir)/lib64" ] &&  export LD_LIBRARY_PATH=$($dep-idir)/lib64:$LD_LIBRARY_PATH
    done

    assimp-
    export LD_LIBRARY_PATH=$(assimp-prefix)/lib:$LD_LIBRARY_PATH
}

ok-linux-install-external() {
    local edeps="glew glfw imgui glm assimp"
    local edep
    for edep in $edeps
    do
        ${edep}-
        ${edep}-get
        ${edep}--
    done
}
ok-linux-install() {

    local deps="bcfg bregex npy ggeo assimpwrap ppm oglrap cudawrap optixrap thrustrap"
    local dep

    for dep in $deps
    do
        $dep-
        $dep--
    done
}



