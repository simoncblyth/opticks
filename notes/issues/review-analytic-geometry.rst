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

     63556 lMaskVirtual_phys0x4c9a510          22      <<< this was csgskiplv : could that cause a problem ?
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


