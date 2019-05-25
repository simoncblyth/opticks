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


optimization ideas 
---------------------

* currently the transforms are taking up quite a lot of buffer realestate, 
  for several reasons

   1. T,V,Q : expect could just use T,V 
   2. suspect there is transform duplication that could be exploited by 
      treating all transforms from all volumes in the MergedMesh assembly
      of volumes together : rather than the current way to dealing with 
      each volume separately 
   3. often the transforms are just translations, even just z-shifts, 
      but still they occupy 4x4x3  

* many node types have quite a lot floats to spare (check this), 
  could a three float translation and a flag to indicate that it is being 
  used avoid referencing to separate transform buffer 

  * historically the space was from when the bbox was stored with the node


* combining sibling instances : in3 and in4 could be combined into one instance
* suspect some of the 201 global volumes must be instanceable



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

     63556 lMaskVirtual_phys0x4c9a510          22      csgskipped
     63557 pMask0x4c3bf20                      17 *   7 parts : difference of two ellipsoid cylinder unions 

     63558 PMT_20inch_log_phys0x4ca16b0        21 *   7 parts : union of el+co+cy  (5 parts, but seven as complete tree)
     63559 PMT_20inch_body_phys0x4c9a7f0       20 *   7 parts : union of el+co+cy  (ditto)
                 
     63560 PMT_20inch_inner1_phys0x4c9a870     18 *   1 part  : el                               cathode vacuum cap
     63561 PMT_20inch_inner2_phys0x4c9a920     19 *   7 parts : union of el+co+cy  (ditto)       remainder vacuum 
                                                   -----------------------------------
                                                      29 parts 
                                                   ------------------------------------

        
     22,17,21,20,19 

     In geocache-j1808-v4-t1  try --csgskiplv 22,17,20,18,19   ## leaving just 21
     In geocache-j1808-v4-t5  try --csgskiplv 22,17,21,20,19   ## leaving just 18 : the cathode cap
     In geocache-j1808-v4-t6  try --csgskiplv 22,17,21,20,18   ## leaving just 19 : vacuum remainder 
     In geocache-j1808-v4-t7  try --csgskiplv 22,17,21,20      ## leaving just 18,19 : vacuum cap+remainder 
     In geocache-j1808-v4-t8  try --csgskiplv 22,17,20         ## leaving just 21,18,19 : outer-pyrex+vacuum cap+remainder 

                                               
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




with the complicated mm2, 1 week later
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* the near degeneracy may well be a problem for validity, but it seems not for performance
  after removing WITH_TORUS


R1/R0 now 3x faster rather than 3x slower::

    ---  METACOMMAND : geocache-bench --xanalytic --enabledmergedmesh 2  GEOFUNC : geocache-j1808-v4 
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558753034 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
                    20190525_105714  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.050      1.000      0.338           1.592    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190525_105714  
                          R1_TITAN_V      0.065      1.302      0.440           1.677    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190525_105714  
            R0_TITAN_V_AND_TITAN_RTX      0.080      1.599      0.540           1.698    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190525_105714  
                          R0_TITAN_V      0.129      2.590      0.876           1.039    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190525_105714  
                        R0_TITAN_RTX      0.148      2.958      1.000           1.070    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190525_105714  
    Namespace(digest=None, exclude=None, include=None, metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='/tmp/blyth/location/results', since=None)

Stable timings::

    ---  METACOMMAND : geocache-bench --xanalytic --enabledmergedmesh 2  GEOFUNC : geocache-j1808-v4 
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558753306 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
                    20190525_110146  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.049      1.000      0.331           0.494    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190525_110146  
                          R1_TITAN_V      0.065      1.321      0.437           0.454    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190525_110146  
            R0_TITAN_V_AND_TITAN_RTX      0.080      1.629      0.539           1.749    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190525_110146  
                          R0_TITAN_V      0.129      2.618      0.867           1.084    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190525_110146  
                        R0_TITAN_RTX      0.148      3.020      1.000           1.001    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190525_110146  
    Namespace(digest=None, exclude=None, include=None, metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='/tmp/blyth/location/results', since=None)



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


Exercise improved digest+geocache handling for easier jumping between geometries
----------------------------------------------------------------------------------

Setup back functions for changing 20inch PMT csgskiplv::

    geocache-export()
    {
        local geofunc=$1
        export OPTICKS_GEOFUNC=$geofunc
        export OPTICKS_KEY=$(${geofunc}-key)
        export OPTICKS_COMMENT=$(${geofunc}-comment)

        geocache-desc
    }
    geocache-desc()
    {
        printf "%-16s : %s \n" "OPTICKS_GEOFUNC" $OPTICKS_GEOFUNC
        printf "%-16s : %s \n" "OPTICKS_KEY"     $OPTICKS_KEY
        printf "%-16s : %s \n" "OPTICKS_COMMENT" $OPTICKS_COMMENT
    }


    geocache-j1808-v4-comment(){ echo reproduce-rtx-inversion-skipping-just-lv-22-maskVirtual ; }
    geocache-j1808-v4-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce ; }
    geocache-j1808-v4-export(){  geocache-export ${FUNCNAME/-export} ; }
    geocache-j1808-v4(){  geocache-j1808-v4- --csgskiplv 22 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) ; }   

    geocache-j1808-v4-t1-comment(){ echo leave-just-21-see-notes/issues/review-analytic-geometry.rst ; }
    geocache-j1808-v4-t1-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.5cc3de75a98f405a4e483bad34be348f ; }
    geocache-j1808-v4-t1-export(){  geocache-export ${FUNCNAME/-export} ; }
    geocache-j1808-v4-t1(){ geocache-j1808-v4- --csgskiplv 22,17,20,18,19 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment)  ; } 

    geocache-j1808-v4-t2-comment(){ echo skip-22-virtualMask+20-almost-degenerate-inner-pyrex-see-notes/issues/review-analytic-geometry.rst ; }
    geocache-j1808-v4-t2-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.781dc285412368f18465809232634d52 ; }
    geocache-j1808-v4-t2-export(){  geocache-export ${FUNCNAME/-export} ; }
    geocache-j1808-v4-t2(){ geocache-j1808-v4- --csgskiplv 22,20 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment)  ; } 

    geocache-j1808-v4-t3-comment(){ echo skip-22-virtualMask+17-mask+20-almost-degenerate-inner-pyrex-see-notes/issues/review-analytic-geometry.rst ; }
    geocache-j1808-v4-t3-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec ; }
    geocache-j1808-v4-t3-export(){  geocache-export ${FUNCNAME/-export} ; }
    geocache-j1808-v4-t3(){ geocache-j1808-v4- --csgskiplv 22,17,20 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment)  ; } 

    geocache-j1808-v4-t4-comment(){ echo skip-22-virtualMask+17-mask+20-almost-degenerate-inner-pyrex+19-remainder-vacuum-see-notes/issues/review-analytic-geometry.rst ; }
    geocache-j1808-v4-t4-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.078714e5894f31953fc9afce731c77f3 ; }
    geocache-j1808-v4-t4-export(){  geocache-export ${FUNCNAME/-export} ; }
    geocache-j1808-v4-t4(){ geocache-j1808-v4- --csgskiplv 22,17,20,19 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment)  ; } 

    geocache-j1808-v4-t5-comment(){ echo just-18-hemi-ellipsoid-cathode-cap-see-notes/issues/review-analytic-geometry.rst ; }
    geocache-j1808-v4-t5-key(){     echo OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.732c52dd2f92338b4c570163ede44230 ; }
    geocache-j1808-v4-t5-export(){  geocache-export ${FUNCNAME/-export} ; }
    geocache-j1808-v4-t5(){ geocache-j1808-v4- --csgskiplv 22,17,21,20,19 --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment)  ; } 

    geocache-bashrc-export(){   geocache-j1808-v4-t5-export ; }




Tee up the geocache via OPTICKS_KEY::

    [blyth@localhost 1]$ geocache-;geocache-j1808-v4-t1-export
              OPTICKS_GEOFUNC : geocache-j1808-v4-t1 
                  OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.5cc3de75a98f405a4e483bad34be348f 
              OPTICKS_COMMENT : leave-just-21-see-notes/issues/review-analytic-geometry.rst 

::

    geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

    OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558355800 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_203640  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.046      1.000      0.548          12.270 
                         R1_TITAN_V      0.066      1.453      0.796           2.789 
                         R0_TITAN_V      0.078      1.709      0.936           6.343 
                       R0_TITAN_RTX      0.083      1.808      0.991           6.334 
                       R1_TITAN_RTX      0.083      1.824      1.000           2.815 


* reproduces the same good behaviour with PMT unsheathed as seen above 20190519_234100



::

     geocache-gui --enabledmergedmesh 2 
     ## just the simplified 20 inch PMT with only LV:21  with good 30fps raytrace performance



geocache-j1808-v4-export : back to bad behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    [blyth@localhost 1]$ geocache-;geocache-j1808-v4-export
              OPTICKS_GEOFUNC : geocache-j1808-v4 
                  OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce 
              OPTICKS_COMMENT : reproduce-rtx-inversion-skipping-just-lv-22-maskVirtual 
    [blyth@localhost 1]$ 


    geocache-gui --enabledmergedmesh 2   ## back to the shielded PMT 

    geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 


::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558356618 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_205018  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.072      1.000      0.212          12.141 
                       R0_TITAN_RTX      0.121      1.675      0.355           6.296 
                         R0_TITAN_V      0.135      1.875      0.397           6.520 
                         R1_TITAN_V      0.315      4.361      0.924           2.859 
                       R1_TITAN_RTX      0.341      4.721      1.000           3.065 

* bad behavior when just skip 22




geocache-j1808-v4-t2 : looks identical have just skipped the near degenererate 20 as well
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

       OPTICKS_GEOFUNC : geocache-j1808-v4-t2 
       OPTICKS_KEY     : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.781dc285412368f18465809232634d52 
       OPTICKS_COMMENT : skip-22-virtualMask+20-almost-degenerate-inner-pyrex-see-notes/issues/review-analytic-geometry.rst 

       [blyth@localhost 1]$ geocache-gui --enabledmergedmesh 2   ## looks identical to above, impossible to see the near degenerate 20 that is skipped


::
     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558357731 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_210851  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.068      1.000      0.254          12.126 
                       R0_TITAN_RTX      0.111      1.643      0.417           6.187 
                         R0_TITAN_V      0.123      1.814      0.460           6.114 
                         R1_TITAN_V      0.246      3.640      0.923           2.828 
                       R1_TITAN_RTX      0.267      3.944      1.000           2.926 


* skipping the degenerate helps a bit, but its not all of the problem : R1 is factor of 2 slower


cache-j1808-v4-t3 : skip the shield too
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    OPTICKS_GEOFUNC  : geocache-j1808-v4-t3 
    OPTICKS_KEY      : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec 
    OPTICKS_COMMENT  : skip-22-virtualMask+17-mask+20-almost-degenerate-inner-pyrex-see-notes/issues/review-analytic-geometry.rst 
    [blyth@localhost 1]$ 
    [blyth@localhost 1]$ geocache-gui --enabledmergedmesh 2


::

     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558358830 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_212710  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.053      1.000      0.259          12.301 
                       R0_TITAN_RTX      0.097      1.830      0.475           6.302 
                         R0_TITAN_V      0.100      1.891      0.490           6.445 
                         R1_TITAN_V      0.172      3.240      0.840           2.858 
                       R1_TITAN_RTX      0.205      3.857      1.000           2.986 




cache-j1808-v4-t4 : skip the remainder vacuum too 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    OPTICKS_GEOFUNC  : geocache-j1808-v4-t4 
    OPTICKS_KEY      : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.078714e5894f31953fc9afce731c77f3 
    OPTICKS_COMMENT  : skip-22-virtualMask+17-mask+20-almost-degenerate-inner-pyrex+19-remainder-vacuum-see-notes/issues/review-analytic-geometry.rst 
    [blyth@localhost ~]$ 
    [blyth@localhost ~]$ 
    [blyth@localhost ~]$ geocache-gui --enabledmergedmesh 2
    ## smth odd here : do not see the ellipsoid z-cut in the raytrace : it looks like a full ellipsoid, but it looks as expected in the rasterized   


::

     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558359862 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_214422  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.050      1.000      0.374          12.372 
                         R0_TITAN_V      0.088      1.740      0.651           6.410 
                       R0_TITAN_RTX      0.090      1.787      0.668           6.354 
                         R1_TITAN_V      0.109      2.173      0.813           2.835 
                       R1_TITAN_RTX      0.134      2.673      1.000           2.948 



geocache-j1808-v4-t5 : skip all except cathode cap : should be a hemi-ellipsoid : SMOKING GUN : RAYTRACE IS FULL ELLIPSOID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    OPTICKS_GEOFUNC  : geocache-j1808-v4-t5 
    OPTICKS_KEY      : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.732c52dd2f92338b4c570163ede44230 
    OPTICKS_COMMENT  : just-18-hemi-ellipsoid-cathode-cap-see-notes/issues/review-analytic-geometry.rst 
    [blyth@localhost ~]$ 
    [blyth@localhost ~]$ geocache-gui --enabledmergedmesh 2


::

     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558361030 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_220350  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.040      1.000      0.554          12.563 
                         R1_TITAN_V      0.049      1.210      0.670           2.841 
                       R1_TITAN_RTX      0.063      1.558      0.863           2.901 
                         R0_TITAN_V      0.065      1.622      0.899           6.433 
                       R0_TITAN_RTX      0.073      1.805      1.000           6.414 

* no RTX inversion with ellipsoid


FIXED BUG WITH LV 18 : HEMI-ELLIPSOIDS BEING MISTRANSLATED INTO ELLIPSOIDS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* see below for details 

Viz check::

    OPTICKS_GEOFUNC  : geocache-j1808-v4-t5 
    OPTICKS_KEY      : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.732c52dd2f92338b4c570163ede44230 
    OPTICKS_COMMENT  : just-18-hemi-ellipsoid-cathode-cap-see-notes/issues/review-analytic-geometry.rst 


    [blyth@localhost opticks]$ geocache-gui --enabledmergedmesh 2


::

     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558364094 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_225454  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.038      1.000      0.552          12.946 
                         R1_TITAN_V      0.046      1.210      0.668           2.844 
                       R1_TITAN_RTX      0.060      1.598      0.882           3.147 
                         R0_TITAN_V      0.062      1.631      0.901           6.539 
                       R0_TITAN_RTX      0.068      1.811      1.000           6.380 


* no RTX inversion with hemi-ellipsoid either



geocache-j1808-v4-t6 : just 19 the vacuum remainder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    OPTICKS_GEOFUNC  : geocache-j1808-v4-t6 
    OPTICKS_KEY      : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.d4157cb873000b4e19f77654134c3196 
    OPTICKS_COMMENT  : just-19-vacuum-remainder-see-notes/issues/review-analytic-geometry.rst 
    [blyth@localhost ~]$ geocache-gui --enabledmergedmesh 2
    ## expected shape 

::

     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558365269 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_231429  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.039      1.000      0.529          12.416 
                         R1_TITAN_V      0.051      1.331      0.704           2.809 
                         R0_TITAN_V      0.066      1.716      0.907           6.476 
                       R1_TITAN_RTX      0.067      1.737      0.919           2.947 
                       R0_TITAN_RTX      0.073      1.891      1.000           6.728 

* fast, no RTX inversion


geocache-j1808-v4-t7 : just 18,19 the vacuum cap+remainder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

    OPTICKS_GEOFUNC  : geocache-j1808-v4-t7 
    OPTICKS_KEY      : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.e13cbdbe8782ca4ca000b735f0c4d61a 
    OPTICKS_COMMENT  : just-18-19-vacuum-cap-and-remainder-see-notes/issues/review-analytic-geometry.rst 
    [blyth@localhost opticks]$ geocache-gui --enabledmergedmesh 2


::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558365964 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_232604  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.047      1.000      0.541          12.490 
                       R1_TITAN_RTX      0.073      1.557      0.843           2.936 
                         R1_TITAN_V      0.077      1.622      0.878           2.864 
                         R0_TITAN_V      0.085      1.808      0.979           6.395 
                       R0_TITAN_RTX      0.087      1.848      1.000           6.323 


* fast, no RTX inversion
* note there is a coincideent face between the two split hemispheres of the vacuum ellipsoid



geocache-j1808-v4-t8 : just 21,18,19,  outer-pyrex+vacuum-cap+remainder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    OPTICKS_GEOFUNC  : geocache-j1808-v4-t8 
    OPTICKS_KEY      : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec 
    OPTICKS_COMMENT  : just-21-18-19-outer-pyrex+vacuum-cap-and-remainder-see-notes/issues/review-analytic-geometry.rst 
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ geocache-gui --enabledmergedmesh 2



     geocache-;geocache-bench --xanalytic --enabledmergedmesh 2 

::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 0,1 --rtx 0 --runfolder geocache-bench --runstamp 1558367315 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190520_234835  launchAVG      rfast      rslow      prelaunch000 
           R0_TITAN_V_AND_TITAN_RTX      0.055      1.000      0.388          12.250 
                       R0_TITAN_RTX      0.097      1.774      0.687           6.340 
                         R0_TITAN_V      0.101      1.847      0.716           6.315 
                       R1_TITAN_RTX      0.140      2.564      0.994           2.907 
                         R1_TITAN_V      0.141      2.580      1.000           2.786 


* RTX inversion starts

* contrast the R1 numbers between t8 and t7, to see the effect of adding the pyrex

  * R0 : little change
  * R1 : doubled RTX ON times for both V and T-rex

* RTX mode really dislikes tightly contained analytic volumes 



geocache-j1808-v4-t8 : after fixing ellipsoid bug and doing cleaninstall for OptiX 511 build test, the R1 times are drastically faster ???
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* RTX ON : x3.6 TITAN RTX 
* RTX ON : x2.3 TITAN V 

* but there was a full clean install too : and the prelaunch times are high : it is as if 
  some cache was cleared and afterwards things are going better 

  * :doc:`nv-ComputeCache`

* there were other changes the ptx too : need to examine all changes over past 24hrs too,
  doing do revealed the cause :doc:`rtxmode-performance-jumps-by-factor-3-or-4-after-flipping-with-torus-switch-off`

* attempt to get more of a good thing by eradicating almost all .f64 :doc:`oxrap-hunt-for-f64-in-ptx`
  have so far not moved the needle 

* TODO : really eradicate ALL the .f64 using WITH_EXCEPTION to see if going f64-less 
  changes anything 



::

     geocache-bench --xanalytic --enabledmergedmesh 2


     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558516754 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190522_171914  launchAVG      rfast      rslow      prelaunch000 
                       R1_TITAN_RTX      0.033      1.000      0.275           2.024 
                         R1_TITAN_V      0.044      1.332      0.366           2.123 
           R0_TITAN_V_AND_TITAN_RTX      0.065      1.972      0.542          21.124 
                         R0_TITAN_V      0.101      3.080      0.847           9.986 
                       R0_TITAN_RTX      0.120      3.638      1.000          11.311 


Rerun reproduces same numbers::

     geocache-bench --xanalytic --enabledmergedmesh 2

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558517866 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190522_173746  launchAVG      rfast      rslow      prelaunch000 
                       R1_TITAN_RTX      0.033      1.000      0.273           0.665 
                         R1_TITAN_V      0.040      1.220      0.333           0.418 
           R0_TITAN_V_AND_TITAN_RTX      0.065      1.988      0.543          11.575 
                         R0_TITAN_V      0.102      3.102      0.847           5.838 
                       R0_TITAN_RTX      0.120      3.660      1.000           5.933 




geocache-j1808-v4-t8 : create the geocache again putting the bug back : as difficult to believe it had such a big effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yep something else caused the big change, (WITH_TORUS)

* the bug has only a minor performance effect on RTX ON

::

     geocache-bench --xanalytic --enabledmergedmesh 2

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558537209 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190522_230009  launchAVG      rfast      rslow      prelaunch000 
                       R1_TITAN_RTX      0.041      1.000      0.333           2.115 
                         R1_TITAN_V      0.047      1.165      0.388           2.059 
           R0_TITAN_V_AND_TITAN_RTX      0.065      1.586      0.529          12.111 
                         R0_TITAN_V      0.101      2.486      0.829           6.109 
                       R0_TITAN_RTX      0.122      3.000      1.000           6.153 


     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558538135 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
                    20190522_231535  launchAVG      rfast      rslow      prelaunch000 
                       R1_TITAN_RTX      0.040      1.000      0.330           0.507 
                         R1_TITAN_V      0.044      1.094      0.362           0.470 
           R0_TITAN_V_AND_TITAN_RTX      0.063      1.570      0.519          11.464 
                         R0_TITAN_V      0.102      2.521      0.833           5.781 
                       R0_TITAN_RTX      0.122      3.026      1.000           5.914 


*repro check*::

    [blyth@localhost ~]$ bench.py --include xanalytic --digest 52e --since May23
    Namespace(digest='52e', exclude=None, include='xanalytic', metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='$TMP/results', since='May23')
    since : 2019-05-23 00:00:00 

    ---
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558577742 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/52e273e4ad5423fe2fc8aa44bbf055ec/1
                    20190523_101542  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.040      1.000      0.340           0.490    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190523_101542  
                          R1_TITAN_V      0.043      1.077      0.366           0.443    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190523_101542  
            R0_TITAN_V_AND_TITAN_RTX      0.064      1.596      0.542          11.661    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190523_101542  
                          R0_TITAN_V      0.102      2.516      0.855           5.930    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190523_101542  
                        R0_TITAN_RTX      0.119      2.942      1.000           5.956    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190523_101542  
    Namespace(digest='52e', exclude=None, include='xanalytic', metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='$TMP/results', since='May23')


*repro check again* after removing all those f64, no difference::

    ---
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558620718 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/52e273e4ad5423fe2fc8aa44bbf055ec/1
                    20190523_221158  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.037      1.000      0.342           1.580    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190523_221158  
                          R1_TITAN_V      0.045      1.214      0.416           1.501    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190523_221158  
            R0_TITAN_V_AND_TITAN_RTX      0.058      1.550      0.531           2.408    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190523_221158  
                          R0_TITAN_V      0.090      2.425      0.830           1.511    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190523_221158  
                        R0_TITAN_RTX      0.109      2.920      1.000           1.400    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190523_221158  



*changed geometry to remove the hemi ellipsoid bug that has artificially put back*::

* only appreciated a little by TITAN RTX


::
    ---
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558621167 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/52e273e4ad5423fe2fc8aa44bbf055ec/1
                    20190523_221927  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.030      1.000      0.280           1.548    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190523_221927  
                          R1_TITAN_V      0.041      1.358      0.380           1.500    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190523_221927  
            R0_TITAN_V_AND_TITAN_RTX      0.058      1.917      0.537           1.717    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190523_221927  
                          R0_TITAN_V      0.091      3.005      0.841           1.093    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190523_221927  
                        R0_TITAN_RTX      0.109      3.573      1.000           0.996    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190523_221927  
    Namespace(digest='52', exclude=None, include=None, metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='$TMP/results', since='6pm')






geocache-j1808-v4-t8 : LV 21,18,19 : check the bbox, how close are they
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    OPTICKS_GEOFUNC  : geocache-j1808-v4-t8 
    OPTICKS_KEY      : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec 
    OPTICKS_COMMENT  : just-21-18-19-outer-pyrex+vacuum-cap-and-remainder-see-notes/issues/review-analytic-geometry.rst 
    [blyth@localhost optixrap]$ geocache-gui --enabledmergedmesh 2 --dbgmm 2 


::

    2019-05-21 09:33:39.404 WARN  [414826] [OGeo::makeAnalyticGeometry@590] [ verbosity 0 lod 0 mm 2
    2019-05-21 09:33:39.404 INFO  [414826] [OGeo::makeAnalyticGeometry@612] mm 2 pts:  GParts  primflag         flagnodetree numParts   15 numPrim    3
    2019-05-21 09:33:39.404 FATAL [414826] [OGeo::makeAnalyticGeometry@616] dumping as instructed by : --dbgmm 2
    2019-05-21 09:33:39.404 INFO  [414826] [GMergedMesh::dumpVolumesSelected@787] OGeo::makeAnalyticGeometry ce0 gfloat4      0.000      0.000    -77.506    261.507  NumVolumes 6 NumVolumesSelected 0
     count     0 idx     2 ce             gfloat4      0.000      0.000    -77.506    261.507  ni[nf/nv/nidx/pidx]         (864,434,63557,63555) id[nidx,midx,bidx,sidx]            (63557, 21, 20,  0)
     count     1 idx     4 ce             gfloat4      0.000      0.000     89.500    249.000  ni[nf/nv/nidx/pidx]         (336,171,63559,63558) id[nidx,midx,bidx,sidx]            (63559, 18, 22,  0)
     count     2 idx     5 ce             gfloat4      0.000      0.000   -167.006    249.000  ni[nf/nv/nidx/pidx]         (624,314,63560,63558) id[nidx,midx,bidx,sidx]            (63560, 19, 23,  0)
    2019-05-21 09:33:39.404 INFO  [414826] [GParts::fulldump@1463] --dbganalytic/--dbgmm lim 10
    2019-05-21 09:33:39.404 INFO  [414826] [GParts::dump@1481] --dbganalytic/--dbgmm lim 10 pbuf 15,4,4
    2019-05-21 09:33:39.404 INFO  [414826] [GParts::dumpPrimInfo@1252] --dbganalytic/--dbgmm (part_offset, parts_for_prim, tran_offset, plan_offset)  numPrim: 3 ulim: 3
    2019-05-21 09:33:39.404 INFO  [414826] [GParts::dumpPrimInfo@1263]  (   0    7    0    0) 
    2019-05-21 09:33:39.404 INFO  [414826] [GParts::dumpPrimInfo@1263]  (   7    1    3    0) 
    2019-05-21 09:33:39.404 INFO  [414826] [GParts::dumpPrimInfo@1263]  (   8    7    4    0) 
    2019-05-21 09:33:39.404 INFO  [414826] [GParts::dump@1498] GParts::dump ni 15 lim 10 ulim 10
         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000      20 <-bnd        0 <-INDEX    bn Water///Pyrex 
         0.0000      0.0000      0.0000           1 (union) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex)               ## nodeIndex is mis-labelled : now used 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000      20 <-bnd        1 <-INDEX    bn Water///Pyrex 
         0.0000      0.0000      0.0000           1 (union) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000     50.0110 
       -60.0105     60.0105      20 <-bnd        2 <-INDEX    bn Water///Pyrex 
         0.0000      0.0000      0.0000          12 (cylinder) TYPECODE 
         0.0000      0.0000      0.0000           3 (nodeIndex) 

         0.0000      0.0000      0.0000    184.0010 
         0.0000      0.0000      20 <-bnd        3 <-INDEX    bn Water///Pyrex 
         0.0000      0.0000      0.0000           5 (sphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

        49.9910    -22.8321     85.2194     22.8321 
         0.0000      0.0000      20 <-bnd        4 <-INDEX    bn Water///Pyrex 
         0.0000      0.0000      0.0000          15 (cone) TYPECODE 
         0.0000      0.0000      0.0000           2 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000      20 <-bnd        5 <-INDEX    bn Water///Pyrex 
         0.0000      0.0000      0.0000           0 (zero) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000      20 <-bnd        6 <-INDEX    bn Water///Pyrex 
         0.0000      0.0000      0.0000           0 (zero) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000    179.0000 
         0.0000    179.0000      22 <-bnd        7 <-INDEX    bn Pyrex/PMT_20inch_photocathode_logsurf2/PMT_20inch_photocathode_logsurf1/Vacuum 
         0.0000      0.0000      0.0000           7 (zsphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000      23 <-bnd        8 <-INDEX    bn Pyrex//PMT_20inch_mirror_logsurf1/Vacuum 
         0.0000      0.0000      0.0000           1 (union) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000      23 <-bnd        9 <-INDEX    bn Pyrex//PMT_20inch_mirror_logsurf1/Vacuum 
         0.0000      0.0000      0.0000           1 (union) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000     45.0100 
       -57.5100     57.5100      23 <-bnd       10 <-INDEX    bn Pyrex//PMT_20inch_mirror_logsurf1/Vacuum 
         0.0000      0.0000      0.0000          12 (cylinder) TYPECODE 
         0.0000      0.0000      0.0000           3 (nodeIndex) 

         0.0000      0.0000      0.0000    179.0000 
      -179.0000      0.0000      23 <-bnd       11 <-INDEX    bn Pyrex//PMT_20inch_mirror_logsurf1/Vacuum 
         0.0000      0.0000      0.0000           7 (zsphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

        44.9900    -25.2457     83.9938     25.2457 
         0.0000      0.0000      23 <-bnd       12 <-INDEX    bn Pyrex//PMT_20inch_mirror_logsurf1/Vacuum 
         0.0000      0.0000      0.0000          15 (cone) TYPECODE 
         0.0000      0.0000      0.0000           2 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000      23 <-bnd       13 <-INDEX    bn Pyrex//PMT_20inch_mirror_logsurf1/Vacuum 
         0.0000      0.0000      0.0000           0 (zero) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000      23 <-bnd       14 <-INDEX    bn Pyrex//PMT_20inch_mirror_logsurf1/Vacuum 
         0.0000      0.0000      0.0000           0 (zero) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

    2019-05-21 09:33:39.405 INFO  [414826] [GParts::Summary@1270] --dbganalytic/--dbgmm num_parts 15 num_prim 3
     part  0 : node  0 type  1 boundary [ 20] Water///Pyrex  
     part  1 : node  0 type  1 boundary [ 20] Water///Pyrex  
     part  2 : node  3 type 12 boundary [ 20] Water///Pyrex  
     part  3 : node  1 type  5 boundary [ 20] Water///Pyrex  
     part  4 : node  2 type 15 boundary [ 20] Water///Pyrex  
     part  5 : node  0 type  0 boundary [ 20] Water///Pyrex  
     part  6 : node  0 type  0 boundary [ 20] Water///Pyrex  
     part  7 : node  1 type  7 boundary [ 22] Pyrex/PMT_20inch_photocathode_logsurf2/PMT_20inch_photocathode_logsurf1/Vacuum  
     part  8 : node  0 type  1 boundary [ 23] Pyrex//PMT_20inch_mirror_logsurf1/Vacuum  
     part  9 : node  0 type  1 boundary [ 23] Pyrex//PMT_20inch_mirror_logsurf1/Vacuum  
     part 10 : node  3 type 12 boundary [ 23] Pyrex//PMT_20inch_mirror_logsurf1/Vacuum  
     part 11 : node  1 type  7 boundary [ 23] Pyrex//PMT_20inch_mirror_logsurf1/Vacuum  
     part 12 : node  2 type 15 boundary [ 23] Pyrex//PMT_20inch_mirror_logsurf1/Vacuum  
     part 13 : node  0 type  0 boundary [ 23] Pyrex//PMT_20inch_mirror_logsurf1/Vacuum  
     part 14 : node  0 type  0 boundary [ 23] Pyrex//PMT_20inch_mirror_logsurf1/Vacuum  
    2019-05-21 09:33:39.405 INFO  [414826] [NPY<T>::dump@1717] partBuf (15,4,4) 

    (  0)       0.000       0.000       0.000       0.000 
    (  0)       0.000       0.000       0.000       0.000 
    (  0)       0.000       0.000       0.000       0.000 
    (  0)       0.000       0.000       0.000       0.000 
    (  1)       0.000       0.000       0.000       0.000 
    (  1)       0.000       0.000       0.000       0.000 
    (  1)       0.000       0.000       0.000       0.000 
    (  1)       0.000       0.000       0.000       0.000 
    (  2)       0.000       0.000       0.000      50.011 
    (  2)     -60.011      60.011       0.000       0.000 
    (  2)       0.000       0.000       0.000       0.000 
    (  2)       0.000       0.000       0.000       0.000 
    (  3)       0.000       0.000       0.000     184.001 
    (  3)       0.000       0.000       0.000       0.000 
    (  3)       0.000       0.000       0.000       0.000 
    (  3)       0.000       0.000       0.000       0.000 
    (  4)      49.991     -22.832      85.219      22.832 
    (  4)       0.000       0.000       0.000       0.000 
    (  4)       0.000       0.000       0.000       0.000 
    (  4)       0.000       0.000       0.000       0.000 
    (  5)       0.000       0.000       0.000       0.000 
    (  5)       0.000       0.000       0.000       0.000 
    (  5)       0.000       0.000       0.000       0.000 
    (  5)       0.000       0.000       0.000       0.000 
    (  6)       0.000       0.000       0.000       0.000 
    (  6)       0.000       0.000       0.000       0.000 
    (  6)       0.000       0.000       0.000       0.000 
    (  6)       0.000       0.000       0.000       0.000 
    (  7)       0.000       0.000       0.000     179.000 
    (  7)       0.000     179.000       0.000       0.000 
    (  7)       0.000       0.000       0.000       0.000 
    (  7)       0.000       0.000       0.000       0.000 
    (  8)       0.000       0.000       0.000       0.000 
    (  8)       0.000       0.000       0.000       0.000 
    (  8)       0.000       0.000       0.000       0.000 
    (  8)       0.000       0.000       0.000       0.000 
    (  9)       0.000       0.000       0.000       0.000 
    (  9)       0.000       0.000       0.000       0.000 
    (  9)       0.000       0.000       0.000       0.000 
    (  9)       0.000       0.000       0.000       0.000 
    ( 10)       0.000       0.000       0.000      45.010 
    ( 10)     -57.510      57.510       0.000       0.000 
    ( 10)       0.000       0.000       0.000       0.000 
    ( 10)       0.000       0.000       0.000       0.000 
    ( 11)       0.000       0.000       0.000     179.000 
    ( 11)    -179.000       0.000       0.000       0.000 
    ( 11)       0.000       0.000       0.000       0.000 
    ( 11)       0.000       0.000       0.000       0.000 
    ( 12)      44.990     -25.246      83.994      25.246 
    ( 12)       0.000       0.000       0.000       0.000 
    ( 12)       0.000       0.000       0.000       0.000 
    ( 12)       0.000       0.000       0.000       0.000 
    ( 13)       0.000       0.000       0.000       0.000 
    ( 13)       0.000       0.000       0.000       0.000 
    ( 13)       0.000       0.000       0.000       0.000 
    ( 13)       0.000       0.000       0.000       0.000 
    ( 14)       0.000       0.000       0.000       0.000 
    ( 14)       0.000       0.000       0.000       0.000 
    ( 14)       0.000       0.000       0.000       0.000 
    ( 14)       0.000       0.000       0.000       0.000 
    2019-05-21 09:33:39.406 INFO  [414826] [NPY<T>::dump@1717] primBuf:partOffset/numParts/primIndex/0 (3,4) 

    (  0)           0           7           0           0 
    (  1)           7           1           3           0 
    (  2)           8           7           4           0 
    2019-05-21 09:33:39.406 FATAL [414826] [OGeo::makeAnalyticGeometry@644]  NodeTree : MISMATCH (numPrim != numVolumes)  numVolumes 6 numVolumesSelected 0 numPrim 3 numPart 15 numTran 7 numPlan 0
    2019-05-21 09:33:39.410 INFO  [414826] [OGeo::makeAnalyticGeometry@712] ] verbosity 0 mm 2



Hmm bounds being calculated::

    115 RT_PROGRAM void bounds (int primIdx, float result[6])
    116 {
    117     //if(primIdx == 0) transform_test();
    118     //if(primIdx == 0) solve_callable_test();
    119 
    120     if(primIdx == 0)
    121     {
    122         unsigned partBuffer_size = partBuffer.size() ;
    123         unsigned planBuffer_size = planBuffer.size() ;
    124         unsigned tranBuffer_size = tranBuffer.size() ;
    125 
    126         rtPrintf("// intersect_analytic.cu:bounds buffer sizes pts:%4d pln:%4d trs:%4d \n", partBuffer_size, planBuffer_size, tranBuffer_size );
    127     }
    128 
    129 
    130     optix::Aabb* aabb = (optix::Aabb*)result;
    131     *aabb = optix::Aabb();
    132 
    133     uint4 identity = identityBuffer[instance_index] ;  // instance_index from OGeo is 0 for non-instanced
    134 
    135     const Prim& prim    = primBuffer[primIdx];
    136 
    137     unsigned primFlag    = prim.primFlag() ;
    138     unsigned partOffset  = prim.partOffset() ;
    139     unsigned numParts    = prim.numParts() ;
    140 
    141 
    142     if(primFlag == CSG_FLAGNODETREE || primFlag == CSG_FLAGINVISIBLE )
    143     {
    144         // identity not strictly needed for bounds, but repeating whats done in intersect for debug convenience
    145         Part pt0 = partBuffer[partOffset + 0] ;
    146         unsigned typecode0 = pt0.typecode() ;
    147         unsigned boundary0 = pt0.boundary() ;
    148 
    149         csg_bounds_prim(primIdx, prim, aabb);
    150 
    151         rtPrintf("// intersect_analytic.cu:bounds.NODETREE primIdx:%2d  bnd0:%3d typ0:%3d "
    152                  " min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n",
    153                     primIdx,
    154                     boundary0,
    155                     typecode0,
    156                     result[0],
    157                     result[1],
    158                     result[2],
    159                     result[3],
    160                     result[4],
    161                     result[5]
    162                 );
    163 
    164     }



::

    geocache-gui --enabledmergedmesh 2 --dbgmm 2 --pindex 1

         ## "--pindex 1" gave rtPrintf output just for the 2nd OptiX primIdx (which is the vacuum-cathode-cap ellipsoid/zsphere)

    geocache-gui --enabledmergedmesh 2 --dbgmm 2 --printenabled --tracer --rtx 1

         ## "--printenabled" as want to see rtPrintf output for all mm2 (20-inch PMT) OptiX primIdx
         ## "--rtx 1" as it initializes faster, with much shorter prelaunch time
         ## "--tracer" as just interested in geometry, not the propagation : this means takes only a few seconds initialization until see geometry

::

    2019-05-21 10:19:35.594 INFO  [30271] [OTracer::trace_@140]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774

    // intersect_analytic.cu:bounds buffer sizes pts:  15 pln:   0 trs:  21 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   0 partOffset   0  numParts   7 -> height  2 -> numNodes  7  tranBuffer_size  21 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   1 partOffset   7  numParts   1 -> height  0 -> numNodes  1  tranBuffer_size  21 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   2 partOffset   8  numParts   7 -> height  2 -> numNodes  7  tranBuffer_size  21 
    // csg_intersect_primitive.h:csg_bounds_sphere  tbb.min (  -254.0010  -254.0010  -184.0010 )  tbb.max (   254.0010   254.0010   184.0010 ) 
    ## csg_bounds_zsphere  zmin   0.000 zmax 179.000  
    ## csg_bounds_zsphere  zmin -179.000 zmax   0.000  
    ## csg_bounds_cone r1:    49.991 z1:   -22.832 r2:    85.219 z2:    22.832 rmax:    85.219 tan_theta:     0.771 z_apex:   -87.632  
    ## csg_bounds_cone r1:    44.990 z1:   -25.246 r2:    83.994 z2:    25.246 rmax:    83.994 tan_theta:     0.772 z_apex:   -83.486  
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius  50.011 z1 -60.011 z2  60.011 
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius  45.010 z1 -57.510 z2  57.510 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 0  bnd0: 20 typ0:  1  min  -254.0010  -254.0010  -339.0110 max   254.0010   254.0010   184.0010     ## 5mm bigger containing bbox union (ellipsoid,cone,cylinder)
    // intersect_analytic.cu:bounds.NODETREE primIdx: 1  bnd0: 22 typ0:  7  min  -249.0000  -249.0000     0.0000 max   249.0000   249.0000   179.0000     ## bbox of upper hemi-ellipsoid 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 2  bnd0: 23 typ0:  1  min  -249.0000  -249.0000  -334.0100 max   249.0000   249.0000     0.0000     ## bbox of union (lower hemi-ellipsoid,cone,cylinder)






check the transforms : how many are used 
-----------------------------------------

::

    147 static __device__
    148 void csg_intersect_part(const Prim& prim, const unsigned partIdx, const float& tt_min, float4& tt  )
    149 {
    150     unsigned tranOffset = prim.tranOffset();
    151     unsigned planOffset = prim.planOffset();
    152     Part pt = partBuffer[partIdx] ;
    153 
    154     unsigned typecode = pt.typecode() ;
    155     unsigned gtransformIdx = pt.gtransformIdx() ;  //  gtransformIdx is 1-based, 0 meaning None
    156     bool complement = pt.complement();
    157 
    158     bool valid_intersect = false ;
    159 
    160     if(gtransformIdx == 0)
    161     {
    162         switch(typecode)


Part.h::

     05 struct Part 
      6 {
      7 
      8     quad q0 ; 
      9     quad q1 ;
     10     quad q2 ;
     11     quad q3 ;
     12 
     13     __device__ unsigned gtransformIdx() const { return q3.u.w & 0x7fffffff ; }  //  gtransformIdx is 1-based, 0 meaning None 
     14     __device__ bool        complement() const { return q3.u.w & 0x80000000 ; }
     15 



geocache-j1808-v4 : back to original, check total number of transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    geocache-gui --enabledmergedmesh 2 --dbgmm 2 --printenabled --tracer --rtx 1


::

    2019-05-21 11:29:46.234 INFO  [143946] [OTracer::trace_@140]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.6602,0.5064,0.5547
    // intersect_analytic.cu:bounds buffer sizes pts:  29 pln:   0 trs:  42 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   0 partOffset   0  numParts   7 -> height  2 -> numNodes  7  tranBuffer_size  42 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   1 partOffset   7  numParts   7 -> height  2 -> numNodes  7  tranBuffer_size  42 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   2 partOffset  14  numParts   7 -> height  2 -> numNodes  7  tranBuffer_size  42 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   3 partOffset  21  numParts   1 -> height  0 -> numNodes  1  tranBuffer_size  42 
    //csg_bounds_prim CSG_FLAGNODETREE  primIdx   4 partOffset  22  numParts   7 -> height  2 -> numNodes  7  tranBuffer_size  42 
    ## csg_bounds_zsphere  zmin   0.000 zmax 179.000  
    ## csg_bounds_zsphere  zmin -179.000 zmax   0.000  
    // csg_intersect_primitive.h:csg_bounds_sphere  tbb.min (  -264.0000  -264.0000  -196.0000 )  tbb.max (   264.0000   264.0000   196.0000 ) 
    // csg_intersect_primitive.h:csg_bounds_sphere  tbb.min (  -254.0010  -254.0010  -184.0010 )  tbb.max (   254.0010   254.0010   184.0010 ) 
    // csg_intersect_primitive.h:csg_bounds_sphere  tbb.min (  -254.0000  -254.0000  -184.0000 )  tbb.max (   254.0000   254.0000   184.0000 ) 
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius 264.000 z1 -177.000 z2 177.000 
    ## csg_bounds_cone r1:    49.991 z1:   -22.832 r2:    85.219 z2:    22.832 rmax:    85.219 tan_theta:     0.771 z_apex:   -87.632  
    ## csg_bounds_cone r1:    49.990 z1:   -22.833 r2:    85.219 z2:    22.833 rmax:    85.219 tan_theta:     0.771 z_apex:   -87.631  
    ## csg_bounds_cone r1:    44.990 z1:   -25.246 r2:    83.994 z2:    25.246 rmax:    83.994 tan_theta:     0.772 z_apex:   -83.486  
    // csg_intersect_primitive.h:csg_bounds_sphere  tbb.min (  -256.0000  -256.0000  -186.0000 )  tbb.max (   256.0000   256.0000   186.0000 ) 
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius 256.000 z1 -172.000 z2 172.000 
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius  50.011 z1 -60.011 z2  60.011 
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius  50.010 z1 -60.010 z2  60.010 
    ## csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius  45.010 z1 -57.510 z2  57.510 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 0  bnd0: 15 typ0:  3  min  -264.0000  -264.0000  -353.9000 max   264.0000   264.0000   196.0000 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 1  bnd0: 20 typ0:  1  min  -254.0010  -254.0010  -339.0110 max   254.0010   254.0010   184.0010 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 2  bnd0: 21 typ0:  1  min  -254.0000  -254.0000  -339.0100 max   254.0000   254.0000   184.0000 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 3  bnd0: 22 typ0:  7  min  -249.0000  -249.0000     0.0000 max   249.0000   249.0000   179.0000 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 4  bnd0: 23 typ0:  1  min  -249.0000  -249.0000  -334.0100 max   249.0000   249.0000     0.0000 






LV 18 : why is the raytrace ellipsoid uncut ? FIXED BUG WITH TRANSLATION OF HEMI-ELLIPSOIDS NOT BECOMING ZSPHERE
------------------------------------------------------------------------------------------------------------------

::

    [blyth@localhost ~]$ CTreeJUNOTest -18
    torus->t mat4
         1.000      0.000      0.000      0.000 
         0.000      1.000      0.000      0.000 
         0.000      0.000      1.000   -219.000 
         0.000      0.000      0.000      1.000 
               tla  vec3       0.000      0.000   -219.000  
    2019-05-20 22:07:24.827 INFO  [239139] [NTreeJUNO::replacement_cone@63]  torus_rhs dvec2(97.000000, -219.000000)
    2019-05-20 22:07:24.827 INFO  [239139] [NTreeJUNO::replacement_cone@66] torus R 97 r 52.01
    2019-05-20 22:07:24.827 INFO  [239139] [nnode::reconstruct_ellipsoid@1892]  sx 1.39106 sy 1.39106 sz 1 radius 179
    2019-05-20 22:07:24.827 INFO  [239139] [NTreeJUNO::replacement_cone@73]  ellipsoid e_axes vec3(249.000000, 249.000000, 179.000000) e_zcut vec2(-179.000000, 179.000000)
    2019-05-20 22:07:24.939 INFO  [239139] [NTreeJUNO::replacement_cone@78]  ca dvec2(83.993834, -168.508521)
    2019-05-20 22:07:24.939 INFO  [239139] [NTreeJUNO::rationalize@115] 
    2019-05-20 22:07:24.939 INFO  [239139] [NTreeJUNO::rationalize@143]  label PMT_20inch_inner1_solid0x4cb3610 is_x018 1 is_x019 0 is_x020 0 is_x021 0
    2019-05-20 22:07:24.939 FATAL [239139] [test_lv@30] LV=-18 label PMT_20inch_inner1_solid0x4cb3610
    2019-05-20 22:07:24.939 ERROR [239139] [test_lv@31] NTreeAnalyse height 0 count 1
      zs


    inorder (left-to-right) 
     [ 0:zs] P PMT_20inch_inner1_solid0x4cb3610 


    2019-05-20 22:07:24.939 INFO  [239139] [nnode::reconstruct_ellipsoid@1892]  sx 1.39106 sy 1.39106 sz 1 radius 179
    G4GDML: Writing solids...
    2019-05-20 22:07:24.942 FATAL [239139] [test_lv@39] <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="SchemaLocation">

      <solids>
        <ellipsoid ax="249" by="249" cz="179" lunit="mm" name="PMT_20inch_inner1_solid0x4cb3610" zcut1="0" zcut2="179"/>
      </solids>

    </gdml>

    2019-05-20 22:07:24.943 INFO  [239139] [test_lv@42] writing gdml to /tmp/blyth/location/CTreeJUNOTest/n018.gdml
    G4GDML: Writing solids...


::

    geocache-;geocache-gui --enabledmergedmesh 2 --dbgmm 2


Add some ellipsoid debug in x4 and recreate::

     geocache-j1808-v4-t5 --dbglv 18

::

    2019-05-20 22:35:02.887 INFO  [286941] [X4PhysicalVolume::convertSolid@488]  [  --dbglv 18 PMT_20inch_inner1_log0x4cb3cc0
    2019-05-20 22:35:02.887 INFO  [286941] [X4Solid::convertEllipsoid@907]  zcut1 0 zcut2 179 z1 -179 z2 179 cz 179 zslice 0
    G4GDML: Writing solids...
    2019-05-20 22:35:02.887 INFO  [286941] [X4PhysicalVolume::convertSolid@501] [--g4codegen] lvIdx 18 soIdx 18 lvname PMT_20inch_inner1_log0x4cb3cc0
    // start portion generated by nnode::to_g4code 
    G4VSolid* make_solid()
    { 
        G4VSolid* a = new G4Ellipsoid("PMT_20inch_inner1_solid0x4cb3610", 249.000000, 249.000000, 179.000000, 0.000000, 179.000000) ; // 0
        return a ; 
    } 
    // end portion generated by nnode::to_g4code 
    G4GDML: Writing solids...
    2019-05-20 22:35:02.888 INFO  [286941] [X4Solid::convertEllipsoid@907]  zcut1 0 zcut2 179 z1 -179 z2 179 cz 179 zslice 0
    2019-05-20 22:35:02.890 INFO  [286941] [NTreeProcess<T>::Process@39] before
    NTreeAnalyse height 0 count 1
      sp


    inorder (left-to-right) 
     [ 0:sp] P PMT_20inch_inner1_solid0x4cb3610_ellipsoid 


    2019-05-20 22:35:02.890 INFO  [286941] [NTreeProcess<T>::Process@54] after
    NTreeAnalyse height 0 count 1
      sp


    inorder (left-to-right) 
     [ 0:sp] P PMT_20inch_inner1_solid0x4cb3610_ellipsoid 


    2019-05-20 22:35:02.890 INFO  [286941] [NTreeProcess<T>::Process@55]  soIdx 18 lvIdx 18 height0 0 height1 0 ### LISTED
    2019-05-20 22:35:02.891 INFO  [286941] [X4PhysicalVolume::convertSolid@532]  ] 18


::

     882 void X4Solid::convertEllipsoid()
     883 {
     884     const G4Ellipsoid* const solid = static_cast<const G4Ellipsoid*>(m_solid);
     885     assert(solid);
     886 
     887     // G4GDMLWriteSolids::EllipsoidWrite
     888 
     889     float ax = solid->GetSemiAxisMax(0)/mm ;
     890     float by = solid->GetSemiAxisMax(1)/mm ;
     891     float cz = solid->GetSemiAxisMax(2)/mm ;
     892 
     893     glm::vec3 scale( ax/cz, by/cz, 1.f) ;
     894     // unity scaling in z, so z-coords are unaffected  
     895 
     896     float zcut1 = solid->GetZBottomCut()/mm ;
     897     float zcut2 = solid->GetZTopCut()/mm ;
     898 
     899 
     900 
     901     float z1 = zcut1 != 0.f && zcut1 > -cz ? zcut1 : -cz ;
     902     float z2 = zcut2 != 0.f && zcut2 <  cz ? zcut2 :  cz ;
     ///                ^^^^^^^^^^^^ WHATS SPECIAL ABOUT ZERO ???

     903     assert( z2 > z1 ) ;
     904 
     905     bool zslice = z1 > -cz || z2 < cz ;
     906 
     907     LOG(info)
     908          << " zcut1 " << zcut1
     909          << " zcut2 " << zcut2
     910          << " z1 " << z1
     911          << " z2 " << z2
     912          << " cz " << cz
     913          << " zslice " << zslice
     914          ;
     915 
     916 
     917     nnode* cn = zslice ?
     918                           (nnode*)make_zsphere( 0.f, 0.f, 0.f, cz, z1, z2 )
     919                        :
     920                           (nnode*)make_sphere( 0.f, 0.f, 0.f, cz )
     921                        ;


