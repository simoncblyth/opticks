review-analytic-geometry
=========================





===============   =================  ================
mm index            gui label          notes
===============   =================  ================
   0                                   global non-instanced
   1                  in0              small PMT
   2                  in1              large PMT
   3                  in2              some TT plate, that manages to be 130 volumes 
   4                  in3              support stick
   5                  in4              support temple
===============   =================  ================




geocache-;geocache-360  whilst looking into geometry
------------------------------------------------------



::

    2019-05-19 20:03:41.782 INFO  [300826] [OGeo::convertMergedMesh@235] ( 0
    2019-05-19 20:03:41.782 ERROR [300826] [OGeo::convertMergedMesh@249]  not converting mesh 0 is_null 0 is_skip 1 is_empty 0


mm1 : small PMT
--------------------

::

    geocache-;geocache-360 --dbgmm 1  


    2019-05-19 20:03:41.782 INFO  [300826] [OGeo::convertMergedMesh@235] ( 1
    2019-05-19 20:03:41.782 INFO  [300826] [OGeo::makeRepeatedAssembly@299]  mmidx 1 imodulo 0
    2019-05-19 20:03:41.782 INFO  [300826] [OGeo::makeOGeometry@543] ugeocode [A]
    2019-05-19 20:03:41.782 INFO  [300826] [OGeo::makeAnalyticGeometry@609] mm 1 pts:  GParts  primflag         flagnodetree numParts    7 numPrim    5
    2019-05-19 20:03:41.787 FATAL [300826] [OGeo::makeOGeometry@565]  DISABLE_ANYHIT 
    2019-05-19 20:03:42.858 INFO  [300826] [OGeo::convertMergedMesh@267] ) 1 numInstances 36572

::

    2019-05-19 20:16:13.417 INFO  [321332] [GMergedMesh::dumpVolumesSelected@787] OGeo::makeAnalyticGeometry ce0 gfloat4      0.002      0.001    -17.937     57.939  NumVolumes 5 NumVolumesSelected 0
     count     0 idx     0 ce             gfloat4      0.002      0.001    -17.937     57.939  ni[nf/nv/nidx/pidx]       (528,266,169989,62592)  id[nidx,midx,bidx,sidx]           (169989, 27, 19,  0)
     count     1 idx     1 ce             gfloat4      0.000      0.000      4.063     39.965  ni[nf/nv/nidx/pidx]       (432,219,169990,169989) id[nidx,midx,bidx,sidx]           (169990, 25, 20,  0)
     count     2 idx     2 ce             gfloat4      0.000      0.000     11.000     36.000  ni[nf/nv/nidx/pidx]       (240,123,169991,169990) id[nidx,midx,bidx,sidx]           (169991, 23, 24,  0)
     count     3 idx     3 ce             gfloat4      0.000      0.000     -4.416     37.812  ni[nf/nv/nidx/pidx]       (288,147,169992,169990) id[nidx,midx,bidx,sidx]           (169992, 24, 25,  0)
     count     4 idx     4 ce             gfloat4      0.000      0.000    -45.874     29.999  ni[nf/nv/nidx/pidx]       ( 96, 50,169993,169989) id[nidx,midx,bidx,sidx]           (169993, 26, 17,  0)
                                                       center                          extent                        faces,verts,node-idx,parent-idx   node-idx/mesh-idx "lv"/boundary/surface
::

    ## seems current geocache doesnt have the codegen ? so look at an old geocache
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/2/g4codegen/tests/x027.cc
             
::

    [blyth@localhost GMeshLib]$ cat MeshUsage.txt    ## reordered to match the above 
    GMeshLib::writeMeshUsage
     meshIndex, nvert, nface, nodeCount, nodeCount*nvert, nodeCount*nface, meshName 
        ...
        27 ( v  266 f  528 ) :   36572 :    9728152 :   19310016 : PMT_3inch_pmt_solid0x510aae0                  : union of sphere and polycone "cylinder" (3 parts)
        25 ( v  219 f  432 ) :   36572 :    8009268 :   15799104 : PMT_3inch_body_solid_ell_ell_helper0x510ada0  : z-cut ellipsoid  (1 part)
        23 ( v  123 f  240 ) :   36572 :    4498356 :    8777280 : PMT_3inch_inner1_solid_ell_helper0x510ae30    : another z-cut ellipsoid (1 part)
        24 ( v  147 f  288 ) :   36572 :    5376084 :   10532736 : PMT_3inch_inner2_solid_ell_helper0x510af10    : yet-another z-cut ellipsoid (1 part)
        26 ( v   50 f   96 ) :   36572 :    1828600 :    3510912 : PMT_3inch_cntr_solid0x510afa0                 : polycone cylinder (1 part)
                                                                                                                 ----------------------------------------
                                                                                                                   expecting 7 parts    : YEP 
                                                                                                                 ----------------------------------------
GNodeLib/PVNames.txt first occurence of 3inch matches the nidx in the above::

    169988 PMT_20inch_inner1_phys0x4c9a870
    169989 PMT_20inch_inner2_phys0x4c9a920

    169990 PMT_3inch_log_phys0x510ddb0               ## NB the number is 1-based from vim   
    169991 PMT_3inch_body_phys0x510be30
    169992 PMT_3inch_inner1_phys0x510beb0
    169993 PMT_3inch_inner2_phys0x510bf60
    169994 PMT_3inch_cntr_phys0x510c010

    169995 PMT_3inch_log_phys0x510de80
    169996 PMT_3inch_body_phys0x510be30
    169997 PMT_3inch_inner1_phys0x510beb0


GNodeLib/LVNames.txt::

    169988 PMT_20inch_inner1_log0x4cb3cc0
    169989 PMT_20inch_inner2_log0x4c9a6e0

    169990 PMT_3inch_log0x510b9f0
    169991 PMT_3inch_body_log0x4bc2650
    169992 PMT_3inch_inner1_log0x510bb00
    169993 PMT_3inch_inner2_log0x510bc10
    169994 PMT_3inch_cntr_log0x510bd20

    169995 PMT_3inch_log0x510b9f0
    169996 PMT_3inch_body_log0x4bc2650
    169997 PMT_3inch_inner1_log0x510bb00
    169998 PMT_3inch_inner2_log0x510bc10
    169999 PMT_3inch_cntr_log0x510bd20




mm2 : large PMT
--------------------

::

    geocache-;geocache-360 --dbgmm 2


    2019-05-19 20:58:32.493 INFO  [390169] [OGeo::convertMergedMesh@235] ( 2
    2019-05-19 20:58:32.493 INFO  [390169] [OGeo::makeRepeatedAssembly@299]  mmidx 2 imodulo 0
    2019-05-19 20:58:32.493 INFO  [390169] [OGeo::makeOGeometry@543] ugeocode [A]
    2019-05-19 20:58:32.493 INFO  [390169] [OGeo::makeAnalyticGeometry@610] mm 2 pts:  GParts  primflag         flagnodetree numParts   29 numPrim    5
    2019-05-19 20:58:32.493 FATAL [390169] [OGeo::makeAnalyticGeometry@615] dumping as instructed by : --dbgmm 2
    2019-05-19 20:58:32.493 INFO  [390169] [GMergedMesh::dumpVolumesSelected@787] OGeo::makeAnalyticGeometry ce0 gfloat4      0.016      0.012    -78.946    274.946  NumVolumes 6 NumVolumesSelected 0
     count     0 idx     1 ce             gfloat4      0.016      0.012    -78.946    274.946  ni[nf/nv/nidx/pidx]         (960,484,63556,63555) id[nidx,midx,bidx,sidx]            (63556, 17, 15,  0)
     count     1 idx     2 ce             gfloat4      0.000      0.000    -77.506    261.507  ni[nf/nv/nidx/pidx]         (864,434,63557,63555) id[nidx,midx,bidx,sidx]            (63557, 21, 20,  0)
     count     2 idx     3 ce             gfloat4      0.000      0.000    -77.506    261.506  ni[nf/nv/nidx/pidx]         (864,434,63558,63557) id[nidx,midx,bidx,sidx]            (63558, 20, 21,  0)
     count     3 idx     4 ce             gfloat4      0.000      0.000     89.500    249.000  ni[nf/nv/nidx/pidx]         (336,171,63559,63558) id[nidx,midx,bidx,sidx]            (63559, 18, 22,  0)
     count     4 idx     5 ce             gfloat4      0.000      0.000   -167.006    249.000  ni[nf/nv/nidx/pidx]         (624,314,63560,63558) id[nidx,midx,bidx,sidx]            (63560, 19, 23,  0)
    2019-05-19 20:58:32.493 FATAL [390169] [OGeo::makeAnalyticGeometry@644]  NodeTree : MISMATCH (numPrim != numVolumes)  numVolumes 6 numVolumesSelected 0 numPrim 5 numPart 29 numTran 14 numPlan 0
    2019-05-19 20:58:32.494 FATAL [390169] [OGeo::makeOGeometry@565]  DISABLE_ANYHIT 
    2019-05-19 20:58:33.084 INFO  [390169] [OGeo::convertMergedMesh@267] ) 2 numInstances 20046



/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/2/g4codegen/tests/x017.cc::

    
    .        di                    7 parts : difference of two ellipsoid cylinder unions
       un           un
    el   cy    el     cy



GNodeLib/PVNames.txt 1-based index from vim, first 20inch::

     63555 lFasteners_phys0x4c31eb0

     63556 lMaskVirtual_phys0x4c9a510          22      <<< this was csgskiplv : could that cause a problem ?
     63557 pMask0x4c3bf20                      17 *   7 parts : difference of two ellipsoid cylinder unions 

     63558 PMT_20inch_log_phys0x4ca16b0        21 *   7 parts : union of el+co+cy  (5 parts, but seven as complete tree)
     63559 PMT_20inch_body_phys0x4c9a7f0       20 *   7 parts : union of el+co+cy  (ditto)
                 
     63560 PMT_20inch_inner1_phys0x4c9a870     18 *   1 part  : el                               cathode vacuum cap
     63561 PMT_20inch_inner2_phys0x4c9a920     19 *   7 parts : union of el+co+cy  (ditto)       remainder vacuum 
                                                   -----------------------------------
                                                      29 parts 
                                                   ------------------------------------

     In geocache-j1808-v4-t1  try --csgskiplv 22,17,20,18,19   ## leaving just 21
                                               
     63562 lMaskVirtual_phys0x4c9a590
     63563 pMask0x4c3bf20
     63564 PMT_20inch_log_phys0x4ca16b0
     63565 PMT_20inch_body_phys0x4c9a7f0

GMeshLib/MeshUsage.txt::

    22 ( v   50 f   96 ) :   20046 :    1002300 :    1924416 : sMask_virtual0x4c36e10

    17 ( v  484 f  960 ) :   20046 :    9702264 :   19244160 : sMask0x4ca38d0
    21 ( v  434 f  864 ) :   20046 :    8699964 :   17319744 : PMT_20inch_pmt_solid0x4c81b40
    20 ( v  434 f  864 ) :   20046 :    8699964 :   17319744 : PMT_20inch_body_solid0x4c90e50
    18 ( v  171 f  336 ) :   20046 :    3427866 :    6735456 : PMT_20inch_inner1_solid0x4cb3610
    19 ( v  314 f  624 ) :   20046 :    6294444 :   12508704 : PMT_20inch_inner2_solid0x4cb3870



* hmm old codegen no use here as removed the torus 
* :doc:`torus_replacement_on_the_fly`

::

    Rationalized GDML snippets for the four solids::

         CTreeJUNOTest -18   : ellipsoid + cone + cylinder


                                un               5 parts, but 7 as complete tree
                          un         cy
                        el  co
                     


         CTreeJUNOTest -19
         CTreeJUNOTest -20
         CTreeJUNOTest -21



opticksdata-jv4-vi::

      1552     <volume name="lMaskVirtual0x4c803b0">
      1553       <materialref ref="Water0x4bb9ba0"/>
      1554       <solidref ref="sMask_virtual0x4c36e10"/>
      1555       <physvol name="pMask0x4c3bf20">
      1556         <volumeref ref="lMask0x4ca3960"/>

          1523     <volume name="lMask0x4ca3960">
          1524       <materialref ref="Acrylic0x4b83450"/>
          1525       <solidref ref="sMask0x4ca38d0"/>
          1526     </volume>

      1557       </physvol>
      1558       <physvol name="PMT_20inch_log_phys0x4ca16b0">                ###### log_phys
      1559         <volumeref ref="PMT_20inch_log0x4cb3bb0"/>

          1545     <volume name="PMT_20inch_log0x4cb3bb0">
          1546       <materialref ref="Pyrex0x4bae2a0"/>
          1547       <solidref ref="PMT_20inch_pmt_solid0x4c81b40"/>           ###### pmt_solid
          1548       <physvol name="PMT_20inch_body_phys0x4c9a7f0">            ###### body_phys
          1549         <volumeref ref="PMT_20inch_body_log0x4cb3aa0"/>

              1535     <volume name="PMT_20inch_body_log0x4cb3aa0">
              1536       <materialref ref="Pyrex0x4bae2a0"/>                   ###### pyrex inside pyrex with almost same dimensions : not healthy 
              1537       <solidref ref="PMT_20inch_body_solid0x4c90e50"/>      ###### body_solid
              1538       <physvol name="PMT_20inch_inner1_phys0x4c9a870">
              1539         <volumeref ref="PMT_20inch_inner1_log0x4cb3cc0"/>

                  1527     <volume name="PMT_20inch_inner1_log0x4cb3cc0">
                  1528       <materialref ref="Vacuum0x4b9b630"/>
                  1529       <solidref ref="PMT_20inch_inner1_solid0x4cb3610"/>
                  1530     </volume>

              1540       </physvol>
              1541       <physvol name="PMT_20inch_inner2_phys0x4c9a920">
              1542         <volumeref ref="PMT_20inch_inner2_log0x4c9a6e0"/>

                  1531     <volume name="PMT_20inch_inner2_log0x4c9a6e0">
                  1532       <materialref ref="Vacuum0x4b9b630"/>
                  1533       <solidref ref="PMT_20inch_inner2_solid0x4cb3870"/>
                  1534     </volume>

              1543       </physvol>
              1544     </volume>

          1550       </physvol>
          1551     </volume>

      1560       </physvol>
      1561     </volume>






SUSPECT NEAR_DEGENERACY OF LV:20 AND LV:21 pyrex inside pyrex AS POTENTIAL PROBLEM
-----------------------------------------------------------------------------------------



with the complicated mm2
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2        ## reproducibility check 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558185148 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190518_211228     metric      rfast      rslow 
           R0_TITAN_V_AND_TITAN_RTX      0.073      1.000      0.217 
                       R0_TITAN_RTX      0.119      1.615      0.350 
                         R0_TITAN_V      0.136      1.859      0.403 
                         R2_TITAN_V      0.314      4.274      0.927 
                         R1_TITAN_V      0.315      4.288      0.930 
                       R1_TITAN_RTX      0.338      4.610      0.999 
                       R2_TITAN_RTX      0.339      4.612      1.000 



test with simplified mm2 : some x4 faster, and RTX does not hinder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

With the geocache-j1808-v4-t1 geometry ie with --csgskiplv 22,17,20,18,19     ## leave just 21, see notes/issues/review-analytic-geometry.rst  
are much faster and RTX does not hinder::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558280460 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190519_234100  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.045      1.000      0.546          24.067 
                         R1_TITAN_V      0.066      1.471      0.803           2.823 
                         R0_TITAN_V      0.078      1.741      0.951          11.123 
                       R1_TITAN_RTX      0.080      1.798      0.981           2.928 
                       R0_TITAN_RTX      0.082      1.832      1.000          13.503 








mm3 : TT plate
-----------------


::

    2019-05-19 20:03:43.436 INFO  [300826] [OGeo::convertMergedMesh@235] ( 3
    2019-05-19 20:03:43.436 INFO  [300826] [OGeo::makeRepeatedAssembly@299]  mmidx 3 imodulo 0
    2019-05-19 20:03:43.436 INFO  [300826] [OGeo::makeOGeometry@543] ugeocode [A]
    2019-05-19 20:03:43.436 INFO  [300826] [OGeo::makeAnalyticGeometry@609] mm 3 pts:  GParts  primflag         flagnodetree numParts  130 numPrim  130
    2019-05-19 20:03:43.436 FATAL [300826] [OGeo::makeOGeometry@565]  DISABLE_ANYHIT 
    2019-05-19 20:03:43.450 INFO  [300826] [OGeo::convertMergedMesh@267] ) 3 numInstances 480


mm4 : support stick
-----------------------

::

    2019-05-19 20:03:43.450 INFO  [300826] [OGeo::convertMergedMesh@235] ( 4
    2019-05-19 20:03:43.450 INFO  [300826] [OGeo::makeRepeatedAssembly@299]  mmidx 4 imodulo 0
    2019-05-19 20:03:43.450 INFO  [300826] [OGeo::makeOGeometry@543] ugeocode [A]
    2019-05-19 20:03:43.450 INFO  [300826] [OGeo::makeAnalyticGeometry@609] mm 4 pts:  GParts  primflag         flagnodetree numParts    3 numPrim    1
    2019-05-19 20:03:43.450 FATAL [300826] [OGeo::makeOGeometry@565]  DISABLE_ANYHIT 
    2019-05-19 20:03:43.464 INFO  [300826] [OGeo::convertMergedMesh@267] ) 4 numInstances 480


mm5 : support temple "fastener"
--------------------------------------

::

    2019-05-19 20:03:43.464 INFO  [300826] [OGeo::convertMergedMesh@235] ( 5
    2019-05-19 20:03:43.464 INFO  [300826] [OGeo::makeRepeatedAssembly@299]  mmidx 5 imodulo 0
    2019-05-19 20:03:43.464 INFO  [300826] [OGeo::makeOGeometry@543] ugeocode [A]
    2019-05-19 20:03:43.464 INFO  [300826] [OGeo::makeAnalyticGeometry@609] mm 5 pts:  GParts  primflag         flagnodetree numParts   31 numPrim    1
    2019-05-19 20:03:43.465 FATAL [300826] [OGeo::makeOGeometry@565]  DISABLE_ANYHIT 
    2019-05-19 20:03:43.479 INFO  [300826] [OGeo::convertMergedMesh@267] ) 5 numInstances 480
    2019-05-19 20:03:43.479 INFO  [300826] [OGeo::convert@230] ] nmm 6




