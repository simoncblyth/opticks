triplet-id-loosing-offset-index-in-NPY
========================================


Cause of the now fixed Issue
-----------------------------

Cause of problem was the old getQuad (now renamed to getQuadF) returning a glm::vec4 float 
which got silently converted into glm::uvec4 : the trouble with that is that everything 
appears as expected until the content value gets sufficiently large that the precision loss
of going through float results in getting wrong values.


FIXED by correcting improper glm::uvec4 getQuad to templated getQuad_
-----------------------------------------------------------------------------

::

    467 glm::uvec4 GNodeLib::getIdentity(unsigned index) const
    468 {
    469     assert( index < m_num_volumes );
    470     //glm::uvec4 id = m_identity->getQuad(index) ; see notes/issues/triplet-id-loosing-offset-index-in-NPY.rst
    471     glm::uvec4 id = m_identity->getQuad_(index) ;
    472     return id ;
    473 }

    752 glm::uvec4 GGeoLib::getIdentity(unsigned ridx, unsigned pidx, unsigned oidx) const
    753 {
    754     checkTriplet(ridx, pidx, oidx);
    755 
    756     GMergedMesh* mm = getMergedMesh(ridx);
    757     assert(mm) ; 
    758     NPY<unsigned>* iib = mm->getInstancedIdentityBuffer();
    759     
    760     //glm::uvec4 id = iib->getQuad(pidx, oidx, 0 );   see notes/issues/triplet-id-loosing-offset-index-in-NPY.rst
    761     glm::uvec4 id = iib->getQuad_(pidx, oidx, 0 ); 
    762     
    763     return id ;
    764 }   



After the fix
----------------


GGeoTest::

    2020-10-13 16:25:45.371 INFO  [8349594] [test_GGeo_getIdentity@232]  ridx 5 num_placements 672 num_volumes 5
     pidx 0
     ridx 5 pidx 0 oidx 0 tid[uvec4(3199, 83886080, 3080219, -1);rpo(5 0 0)    5000000] nid[uvec4(3199, 83886080, 3080219, -1);rpo(5 0 0)    5000000]
     ridx 5 pidx 0 oidx 1 tid[uvec4(3200, 83886081, 3014684, -1);rpo(5 0 1)    5000001] nid[uvec4(3200, 83886081, 3014684, -1);rpo(5 0 1)    5000001]
     ridx 5 pidx 0 oidx 2 tid[uvec4(3201, 83886082, 2818077, -1);rpo(5 0 2)    5000002] nid[uvec4(3201, 83886082, 2818077, -1);rpo(5 0 2)    5000002]
     ridx 5 pidx 0 oidx 3 tid[uvec4(3202, 83886083, 2883614, -1);rpo(5 0 3)    5000003] nid[uvec4(3202, 83886083, 2883614, -1);rpo(5 0 3)    5000003]
     ridx 5 pidx 0 oidx 4 tid[uvec4(3203, 83886084, 2949150, -1);rpo(5 0 4)    5000004] nid[uvec4(3203, 83886084, 2949150, -1);rpo(5 0 4)    5000004]
     pidx 1
     ridx 5 pidx 1 oidx 0 tid[uvec4(3205, 83886336, 3080219, -1);rpo(5 1 0)    5000100] nid[uvec4(3205, 83886336, 3080219, -1);rpo(5 1 0)    5000100]
     ridx 5 pidx 1 oidx 1 tid[uvec4(3206, 83886337, 3014684, -1);rpo(5 1 1)    5000101] nid[uvec4(3206, 83886337, 3014684, -1);rpo(5 1 1)    5000101]
     ridx 5 pidx 1 oidx 2 tid[uvec4(3207, 83886338, 2818077, -1);rpo(5 1 2)    5000102] nid[uvec4(3207, 83886338, 2818077, -1);rpo(5 1 2)    5000102]
     ridx 5 pidx 1 oidx 3 tid[uvec4(3208, 83886339, 2883614, -1);rpo(5 1 3)    5000103] nid[uvec4(3208, 83886339, 2883614, -1);rpo(5 1 3)    5000103]
     ridx 5 pidx 1 oidx 4 tid[uvec4(3209, 83886340, 2949150, -1);rpo(5 1 4)    5000104] nid[uvec4(3209, 83886340, 2949150, -1);rpo(5 1 4)    5000104]
     pidx 2
     ridx 5 pidx 2 oidx 0 tid[uvec4(3211, 83886592, 3080219, -1);rpo(5 2 0)    5000200] nid[uvec4(3211, 83886592, 3080219, -1);rpo(5 2 0)    5000200]
     ridx 5 pidx 2 oidx 1 tid[uvec4(3212, 83886593, 3014684, -1);rpo(5 2 1)    5000201] nid[uvec4(3212, 83886593, 3014684, -1);rpo(5 2 1)    5000201]
     ridx 5 pidx 2 oidx 2 tid[uvec4(3213, 83886594, 2818077, -1);rpo(5 2 2)    5000202] nid[uvec4(3213, 83886594, 2818077, -1);rpo(5 2 2)    5000202]
     ridx 5 pidx 2 oidx 3 tid[uvec4(3214, 83886595, 2883614, -1);rpo(5 2 3)    5000203] nid[uvec4(3214, 83886595, 2883614, -1);rpo(5 2 3)    5000203]
     ridx 5 pidx 2 oidx 4 tid[uvec4(3215, 83886596, 2949150, -1);rpo(5 2 4)    5000204] nid[uvec4(3215, 83886596, 2949150, -1);rpo(5 2 4)    5000204]
    epsilon:opticks blyth$ 


GNodeLibTest::

    2020-10-13 16:26:29.489 INFO  [8350384] [test_getIdentity@90] 
     nidx 3199 nid[uvec4(3199, 83886080, 3080219, -1);rpo(5 0 0)    5000000]
     nidx 3200 nid[uvec4(3200, 83886081, 3014684, -1);rpo(5 0 1)    5000001]
     nidx 3201 nid[uvec4(3201, 83886082, 2818077, -1);rpo(5 0 2)    5000002]
     nidx 3202 nid[uvec4(3202, 83886083, 2883614, -1);rpo(5 0 3)    5000003]
     nidx 3203 nid[uvec4(3203, 83886084, 2949150, -1);rpo(5 0 4)    5000004]
     nidx 3204 nid[uvec4(3204, 1407, 3145759, -1);rpo(0 0 1407)        57f]
     nidx 3205 nid[uvec4(3205, 83886336, 3080219, -1);rpo(5 1 0)    5000100]
     nidx 3206 nid[uvec4(3206, 83886337, 3014684, -1);rpo(5 1 1)    5000101]
     nidx 3207 nid[uvec4(3207, 83886338, 2818077, -1);rpo(5 1 2)    5000102]
     nidx 3208 nid[uvec4(3208, 83886339, 2883614, -1);rpo(5 1 3)    5000103]
     nidx 3209 nid[uvec4(3209, 83886340, 2949150, -1);rpo(5 1 4)    5000104]
     nidx 3210 nid[uvec4(3210, 1408, 3145759, -1);rpo(0 0 1408)        580]
     nidx 3211 nid[uvec4(3211, 83886592, 3080219, -1);rpo(5 2 0)    5000200]
     nidx 3212 nid[uvec4(3212, 83886593, 3014684, -1);rpo(5 2 1)    5000201]
     nidx 3213 nid[uvec4(3213, 83886594, 2818077, -1);rpo(5 2 2)    5000202]
     nidx 3214 nid[uvec4(3214, 83886595, 2883614, -1);rpo(5 2 3)    5000203]
     nidx 3215 nid[uvec4(3215, 83886596, 2949150, -1);rpo(5 2 4)    5000204]
     nidx 3216 nid[uvec4(3216, 1409, 3145759, -1);rpo(0 0 1409)        581]
     nidx 3217 nid[uvec4(3217, 83886848, 3080219, -1);rpo(5 3 0)    5000300]
     nidx 3218 nid[uvec4(3218, 83886849, 3014684, -1);rpo(5 3 1)    5000301]
    epsilon:opticks blyth$ 


numpyTest still demonstrates the issue as a reminder of what not to do::

    epsilon:opticks blyth$ numpyTest 
    2020-10-13 16:27:06.822 INFO  [8350827] [BOpticksKey::SetKey@75]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
    2020-10-13 16:27:06.824 INFO  [8350827] [BOpticksResource::setupViaKey@880] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
                     exename  : OKX4Test
             current_exename  : numpyTest
                       class  : X4PhysicalVolume
                     volname  : World0xc15cfc00x40f7000_PV
                      digest  : 50a18baaf29b18fae8c1642927003ee3
                      idname  : OKX4Test_World0xc15cfc00x40f7000_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-10-13 16:27:06.824 INFO  [8350827] [test_getters@192] 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 16:27:06.824 INFO  [8350827] [numpyTest::numpyTest@33]  aoba::LoadArrayFromNumpy 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 16:27:06.825 INFO  [8350827] [numpyTest::numpyTest@38]  NPY<unsigned>::load, getValuesConst 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 16:27:06.827 INFO  [8350827] [numpyTest::numpyTest@50]  NPY<unsigned>::load, getValue 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 16:27:06.829 INFO  [8350827] [numpyTest::numpyTest@51]  NPY<unsigned>::load, getQuadF 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b         0
    ( 3200)       3200   5000000    2e001c         0
    ( 3201)       3201   5000000    2b001d         0
    ( 3202)       3202   5000000    2c001e         0
    ( 3203)       3203   5000000    2d001e         0
    ( 3204)       3204       57f    30001f         0
    ( 3205)       3205   5000100    2f001b         0
    ( 3206)       3206   5000100    2e001c         0
    ( 3207)       3207   5000100    2b001d         0
    ( 3208)       3208   5000100    2c001e         0
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 16:27:06.833 INFO  [8350827] [numpyTest::numpyTest@52]  NPY<unsigned>::load, getQuad_ 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 16:27:06.836 INFO  [8350827] [numpyTest::numpyTest@53]  NPY<unsigned>::load, getQuadU 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 16:27:06.839 INFO  [8350827] [numpyTest::numpyTest@54]  NPY<unsigned>::load, getQuadI 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 16:27:06.844 INFO  [8350827] [numpyTest::numpyTest@55]  NPY<unsigned>::load, getQuadLocal 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b         0
    ( 3200)       3200   5000000    2e001c         0
    ( 3201)       3201   5000000    2b001d         0
    ( 3202)       3202   5000000    2c001e         0
    ( 3203)       3203   5000000    2d001e         0
    ( 3204)       3204       57f    30001f         0
    ( 3205)       3205   5000100    2f001b         0
    ( 3206)       3206   5000100    2e001c         0
    ( 3207)       3207   5000100    2b001d         0
    ( 3208)       3208   5000100    2c001e         0
    epsilon:opticks blyth$ 




Issue: Triplet id for all 5 volumes of ridx 5 are those of outer volume (oidx:0)
---------------------------------------------------------------------------------------

::


    2020-10-13 08:48:55.277 INFO  [7956120] [test_GGeo_getIdentity@225] 
     pidx 0
     ridx 5 pidx 0 oidx 0 id uvec4(3199, 83886080, 3080219, 0) rpo (5 0 0)
     ridx 5 pidx 0 oidx 1 id uvec4(3200, 83886080, 3014684, 0) rpo (5 0 0)
     ridx 5 pidx 0 oidx 2 id uvec4(3201, 83886080, 2818077, 0) rpo (5 0 0)
     ridx 5 pidx 0 oidx 3 id uvec4(3202, 83886080, 2883614, 0) rpo (5 0 0)
     ridx 5 pidx 0 oidx 4 id uvec4(3203, 83886080, 2949150, 0) rpo (5 0 0)
     pidx 1
     ridx 5 pidx 1 oidx 0 id uvec4(3205, 83886336, 3080219, 0) rpo (5 1 0)
     ridx 5 pidx 1 oidx 1 id uvec4(3206, 83886336, 3014684, 0) rpo (5 1 0)
     ridx 5 pidx 1 oidx 2 id uvec4(3207, 83886336, 2818077, 0) rpo (5 1 0)
     ridx 5 pidx 1 oidx 3 id uvec4(3208, 83886336, 2883614, 0) rpo (5 1 0)
     ridx 5 pidx 1 oidx 4 id uvec4(3209, 83886336, 2949150, 0) rpo (5 1 0)
     pidx 2
     ridx 5 pidx 2 oidx 0 id uvec4(3211, 83886592, 3080219, 0) rpo (5 2 0)
     ridx 5 pidx 2 oidx 1 id uvec4(3212, 83886592, 3014684, 0) rpo (5 2 0)
     ridx 5 pidx 2 oidx 2 id uvec4(3213, 83886592, 2818077, 0) rpo (5 2 0)
     ridx 5 pidx 2 oidx 3 id uvec4(3214, 83886592, 2883614, 0) rpo (5 2 0)
     ridx 5 pidx 2 oidx 4 id uvec4(3215, 83886592, 2949150, 0) rpo (5 2 0)
                                         ^^^^^^^^

Identity from triplet access and nidx access have same problem of triplet identity stuck at that of outer volume::

    2020-10-13 09:33:54.862 INFO  [7992696] [test_GGeo_getIdentity@251]  ridx 5 num_placements 672 num_volumes 5
     pidx 0
     ridx 5 pidx 0 oidx 0 tid[uvec4(3199, 83886080, 3080219, 0) ; rpo (5 0 0)] nid[uvec4(3199, 83886080, 3080219, 0) ; rpo (5 0 0)]
     ridx 5 pidx 0 oidx 1 tid[uvec4(3200, 83886080, 3014684, 0) ; rpo (5 0 0)] nid[uvec4(3200, 83886080, 3014684, 0) ; rpo (5 0 0)]
     ridx 5 pidx 0 oidx 2 tid[uvec4(3201, 83886080, 2818077, 0) ; rpo (5 0 0)] nid[uvec4(3201, 83886080, 2818077, 0) ; rpo (5 0 0)]
     ridx 5 pidx 0 oidx 3 tid[uvec4(3202, 83886080, 2883614, 0) ; rpo (5 0 0)] nid[uvec4(3202, 83886080, 2883614, 0) ; rpo (5 0 0)]
     ridx 5 pidx 0 oidx 4 tid[uvec4(3203, 83886080, 2949150, 0) ; rpo (5 0 0)] nid[uvec4(3203, 83886080, 2949150, 0) ; rpo (5 0 0)]
     pidx 1
     ridx 5 pidx 1 oidx 0 tid[uvec4(3205, 83886336, 3080219, 0) ; rpo (5 1 0)] nid[uvec4(3205, 83886336, 3080219, 0) ; rpo (5 1 0)]
     ridx 5 pidx 1 oidx 1 tid[uvec4(3206, 83886336, 3014684, 0) ; rpo (5 1 0)] nid[uvec4(3206, 83886336, 3014684, 0) ; rpo (5 1 0)]
     ridx 5 pidx 1 oidx 2 tid[uvec4(3207, 83886336, 2818077, 0) ; rpo (5 1 0)] nid[uvec4(3207, 83886336, 2818077, 0) ; rpo (5 1 0)]
     ridx 5 pidx 1 oidx 3 tid[uvec4(3208, 83886336, 2883614, 0) ; rpo (5 1 0)] nid[uvec4(3208, 83886336, 2883614, 0) ; rpo (5 1 0)]
     ridx 5 pidx 1 oidx 4 tid[uvec4(3209, 83886336, 2949150, 0) ; rpo (5 1 0)] nid[uvec4(3209, 83886336, 2949150, 0) ; rpo (5 1 0)]
     pidx 2
     ridx 5 pidx 2 oidx 0 tid[uvec4(3211, 83886592, 3080219, 0) ; rpo (5 2 0)] nid[uvec4(3211, 83886592, 3080219, 0) ; rpo (5 2 0)]
     ridx 5 pidx 2 oidx 1 tid[uvec4(3212, 83886592, 3014684, 0) ; rpo (5 2 0)] nid[uvec4(3212, 83886592, 3014684, 0) ; rpo (5 2 0)]
     ridx 5 pidx 2 oidx 2 tid[uvec4(3213, 83886592, 2818077, 0) ; rpo (5 2 0)] nid[uvec4(3213, 83886592, 2818077, 0) ; rpo (5 2 0)]
     ridx 5 pidx 2 oidx 3 tid[uvec4(3214, 83886592, 2883614, 0) ; rpo (5 2 0)] nid[uvec4(3214, 83886592, 2883614, 0) ; rpo (5 2 0)]
     ridx 5 pidx 2 oidx 4 tid[uvec4(3215, 83886592, 2949150, 0) ; rpo (5 2 0)] nid[uvec4(3215, 83886592, 2949150, 0) ; rpo (5 2 0)]
    epsilon:ggeo blyth$ 


GNodeLibTest shows the same::

    epsilon:ggeo blyth$ NPY=INFO GNodeLibTest 
    2020-10-13 10:05:17.725 INFO  [8022518] [BOpticksKey::SetKey@75]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
    ...
    2020-10-13 10:05:17.730 INFO  [8022518] [NMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "GEOCACHE_CODE_VERSION": 6,
        "argline": "/usr/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v0.gdml --x4polyskip 211,232 --geocenter --noviz --runfolder geocache-dx0 --runcomment export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation.rst -D ",
        "location": "Opticks::updateCacheMeta",
        "runcomment": "export-dyb-gdml-from-g4-10-4-2-to-support-geocache-creation.rst",
        "rundate": "20201012_122022",
        "runfolder": "geocache-dx0",
        "runlabel": "R0_cvd_",
        "runstamp": 1602501622
    }
    2020-10-13 10:05:17.731 INFO  [8022518] [Opticks::loadOriginCacheMeta@1886] ExtractCacheMetaGDMLPath /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v0.gdml
    2020-10-13 10:05:17.731 INFO  [8022518] [Opticks::loadOriginCacheMeta@1914] (pass) GEOCACHE_CODE_VERSION 6
    2020-10-13 10:05:17.818 INFO  [8022518] [test_getIdentity@90] 
     nidx 3199 nid[uvec4(3199, 83886080, 3080219, 0);rpo(5 0 0)    5000000]
     nidx 3200 nid[uvec4(3200, 83886080, 3014684, 0);rpo(5 0 0)    5000000]
     nidx 3201 nid[uvec4(3201, 83886080, 2818077, 0);rpo(5 0 0)    5000000]
     nidx 3202 nid[uvec4(3202, 83886080, 2883614, 0);rpo(5 0 0)    5000000]
     nidx 3203 nid[uvec4(3203, 83886080, 2949150, 0);rpo(5 0 0)    5000000]
     nidx 3204 nid[uvec4(3204, 1407, 3145759, 0);rpo(0 0 1407)        57f]
     nidx 3205 nid[uvec4(3205, 83886336, 3080219, 0);rpo(5 1 0)    5000100]
     nidx 3206 nid[uvec4(3206, 83886336, 3014684, 0);rpo(5 1 0)    5000100]
     nidx 3207 nid[uvec4(3207, 83886336, 2818077, 0);rpo(5 1 0)    5000100]
     nidx 3208 nid[uvec4(3208, 83886336, 2883614, 0);rpo(5 1 0)    5000100]
     nidx 3209 nid[uvec4(3209, 83886336, 2949150, 0);rpo(5 1 0)    5000100]
     nidx 3210 nid[uvec4(3210, 1408, 3145759, 0);rpo(0 0 1408)        580]
     nidx 3211 nid[uvec4(3211, 83886592, 3080219, 0);rpo(5 2 0)    5000200]
     nidx 3212 nid[uvec4(3212, 83886592, 3014684, 0);rpo(5 2 0)    5000200]
     nidx 3213 nid[uvec4(3213, 83886592, 2818077, 0);rpo(5 2 0)    5000200]
     nidx 3214 nid[uvec4(3214, 83886592, 2883614, 0);rpo(5 2 0)    5000200]
     nidx 3215 nid[uvec4(3215, 83886592, 2949150, 0);rpo(5 2 0)    5000200]
     nidx 3216 nid[uvec4(3216, 1409, 3145759, 0);rpo(0 0 1409)        581]
     nidx 3217 nid[uvec4(3217, 83886848, 3080219, 0);rpo(5 3 0)    5000300]
     nidx 3218 nid[uvec4(3218, 83886848, 3014684, 0);rpo(5 3 0)    5000300]
    epsilon:ggeo blyth$ 


From python numpy see that the offsets are there in the files::


    In [16]: np.set_printoptions(formatter={'int':hex})                                                                                                                                               
    In [17]: iid[0]                                                                                                                                                                                   
    Out[17]: 
    array([[0xc7f, 0x5000000, 0x2f001b, 0xffffffff],
           [0xc80, 0x5000001, 0x2e001c, 0xffffffff],
           [0xc81, 0x5000002, 0x2b001d, 0xffffffff],
           [0xc82, 0x5000003, 0x2c001e, 0xffffffff],
           [0xc83, 0x5000004, 0x2d001e, 0xffffffff]], dtype=uint32)

    In [18]: iid[1]                                                                                                                                                                                   
    Out[18]: 
    array([[0xc85, 0x5000100, 0x2f001b, 0xffffffff],
           [0xc86, 0x5000101, 0x2e001c, 0xffffffff],
           [0xc87, 0x5000102, 0x2b001d, 0xffffffff],
           [0xc88, 0x5000103, 0x2c001e, 0xffffffff],
           [0xc89, 0x5000104, 0x2d001e, 0xffffffff]], dtype=uint32)

    In [19]:                                                               

Ditto with xxd dumping the bytes::

     xxd all_volume_identity.npy > all_volume_identity.npy.xxd   
     ## xxd dump with header and 1st line moved to tail, to make vim line numbers correspond to 0-based index 
     ## little-endian byte order : lsb at smaller address in the file ?

     3199 0000c840: 7f0c 0000 0000 0005 1b00 2f00 ffff ffff  ........../.....
     3200 0000c850: 800c 0000 0100 0005 1c00 2e00 ffff ffff  ................
     3201 0000c860: 810c 0000 0200 0005 1d00 2b00 ffff ffff  ..........+.....
     3202 0000c870: 820c 0000 0300 0005 1e00 2c00 ffff ffff  ..........,.....
     3203 0000c880: 830c 0000 0400 0005 1e00 2d00 ffff ffff  ..........-.....

     3204 0000c890: 840c 0000 7f05 0000 1f00 3000 ffff ffff  ..........0.....
     3205 0000c8a0: 850c 0000 0001 0005 1b00 2f00 ffff ffff  ........../.....
     3206 0000c8b0: 860c 0000 0101 0005 1c00 2e00 ffff ffff  ................
     3207 0000c8c0: 870c 0000 0201 0005 1d00 2b00 ffff ffff  ..........+.....
     3208 0000c8d0: 880c 0000 0301 0005 1e00 2c00 ffff ffff  ..........,.....
     3209 0000c8e0: 890c 0000 0401 0005 1e00 2d00 ffff ffff  ..........-.....



All looks ok from python numpy?::

    In [1]: iid = np.load("placement_iidentity.npy")                                                                                                                       
    In [2]: iid.shape                                                                                                                                                      
    Out[2]: (672, 5, 4)

    In [3]: iid                                                                                                                                                            
    Out[3]: 
    array([[[      3199,   83886080,    3080219, 4294967295],
            [      3200,   83886081,    3014684, 4294967295],
            [      3201,   83886082,    2818077, 4294967295],
            [      3202,   83886083,    2883614, 4294967295],
            [      3203,   83886084,    2949150, 4294967295]],

           [[      3205,   83886336,    3080219, 4294967295],
            [      3206,   83886337,    3014684, 4294967295],
            [      3207,   83886338,    2818077, 4294967295],
            [      3208,   83886339,    2883614, 4294967295],
            [      3209,   83886340,    2949150, 4294967295]],

    In [6]: iid[0]                                                                                                                                                         
    Out[6]: 
    array([[      3199,   83886080,    3080219, 4294967295],
           [      3200,   83886081,    3014684, 4294967295],
           [      3201,   83886082,    2818077, 4294967295],
           [      3202,   83886083,    2883614, 4294967295],
           [      3203,   83886084,    2949150, 4294967295]], dtype=uint32)

    In [8]: iid[0,:,1]                                                                                                                                                     
    Out[8]: array([83886080, 83886081, 83886082, 83886083, 83886084], dtype=uint32)

    In [10]: list(map(hex, iid[0,:,1] ))                                                                                                                                   
    Out[10]: ['0x5000000', '0x5000001', '0x5000002', '0x5000003', '0x5000004']





    In [4]: avi = np.load("../../GNodeLib/all_volume_identity.npy")                                                                                                        
    In [5]: avi[3199:3211]                                                                                                                                                 
    Out[5]: 
    array([[      3199,   83886080,    3080219, 4294967295],
           [      3200,   83886081,    3014684, 4294967295],
           [      3201,   83886082,    2818077, 4294967295],
           [      3202,   83886083,    2883614, 4294967295],
           [      3203,   83886084,    2949150, 4294967295],
           [      3204,       1407,    3145759, 4294967295],
           [      3205,   83886336,    3080219, 4294967295],
           [      3206,   83886337,    3014684, 4294967295],
           [      3207,   83886338,    2818077, 4294967295],
           [      3208,   83886339,    2883614, 4294967295],
           [      3209,   83886340,    2949150, 4294967295],
           [      3210,       1408,    3145759, 4294967295]], dtype=uint32)

    In [6]:                             




Low level numpyTest does not loose the offsets, suggesting a NPY bug with large unsigned bitfields::

    epsilon:npy blyth$ om-;TEST=numpyTest om-t
    ...
    2020-10-13 10:55:03.415 INFO  [8077420] [main@12] 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    ( 3209)       3209   5000104    2d001e  ffffffff
    ( 3210)       3210       580    30001f  ffffffff
    ( 3211)       3211   5000200    2f001b  ffffffff
    ( 3212)       3212   5000201    2e001c  ffffffff
    ( 3213)       3213   5000202    2b001d  ffffffff
    ( 3214)       3214   5000203    2c001e  ffffffff
    ( 3215)       3215   5000204    2d001e  ffffffff
    ( 3216)       3216       581    30001f  ffffffff
    ( 3217)       3217   5000300    2f001b  ffffffff
    ( 3218)       3218   5000301    2e001c  ffffffff




Adding different load modes to numpyTest see that the bug is in "NPY::getQuad" and not "NPY::getQuad_"::

    [100%] Built target numpyTest
    2020-10-13 11:24:24.346 INFO  [8095360] [main@104] 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 11:24:24.347 INFO  [8095360] [numpyTest::numpyTest@21]  aoba::LoadArrayFromNumpy 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 11:24:24.347 INFO  [8095360] [numpyTest::numpyTest@26]  NPY<unsigned>::load, getValuesConst 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 11:24:24.349 INFO  [8095360] [numpyTest::numpyTest@38]  NPY<unsigned>::load, getQuad 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b         0
    ( 3200)       3200   5000000    2e001c         0
    ( 3201)       3201   5000000    2b001d         0
    ( 3202)       3202   5000000    2c001e         0
    ( 3203)       3203   5000000    2d001e         0
    ( 3204)       3204       57f    30001f         0
    ( 3205)       3205   5000100    2f001b         0
    ( 3206)       3206   5000100    2e001c         0
    ( 3207)       3207   5000100    2b001d         0
    ( 3208)       3208   5000100    2c001e         0
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_identity.npy
    2020-10-13 11:24:24.352 INFO  [8095360] [numpyTest::numpyTest@39]  NPY<unsigned>::load, getQuad_ 
     shape ( 12230 4  ) 
    ( 3199)       3199   5000000    2f001b  ffffffff
    ( 3200)       3200   5000001    2e001c  ffffffff
    ( 3201)       3201   5000002    2b001d  ffffffff
    ( 3202)       3202   5000003    2c001e  ffffffff
    ( 3203)       3203   5000004    2d001e  ffffffff
    ( 3204)       3204       57f    30001f  ffffffff
    ( 3205)       3205   5000100    2f001b  ffffffff
    ( 3206)       3206   5000101    2e001c  ffffffff
    ( 3207)       3207   5000102    2b001d  ffffffff
    ( 3208)       3208   5000103    2c001e  ffffffff
    epsilon:npy blyth$ 





