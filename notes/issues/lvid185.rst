lvid 185
===========


Issue bbox v.diff
--------------------

Using the composite CSG bbox would avoid false alarms from failed ana polygonization.

::

     1155.6  MOFTTube0xc046b40 lvidx 185 

     amn (    574.598   -29.010  -113.129) 
     bmn (   -581.000  -581.000  -127.500) 
     dmn (   1155.598   551.990    14.371) 

     amx (    580.602    29.010   113.129) 
     bmx (    581.000   581.000   127.500) 
     dmx (     -0.398  -551.990   -14.371)


    NCSG::dump (root) [ 0:di] OPER  v:0
             L [ 1:cy] PRIM  v:0 bb  mi  (-581.00 -581.00 -127.50)  mx  ( 581.00  581.00  127.50)  si  (1162.00 1162.00  255.00) 
             R [ 2:cy] PRIM  v:0 bb  mi  (-575.00 -575.00 -128.77)  mx  ( 575.00  575.00  128.77)  si  (1150.00 1150.00  257.55) 
     composite_bb 
            mi  (-581.00 -581.00 -127.50)  
            mx  (  581.00  581.00  127.50)  si  (1162.00 1162.00  255.00) 



Diagnosis : failed ana polygonization (gltf 3 looks fine)
------------------------------------------------------------

::

     DBGMESH=MOFTTube0xc046b40 NSceneMeshTest 

     opticks-tbool 185     # 
     opticks-tbool-vi 185  # 

     op --dlv185         
     op --dlv185 --gltf 1  # ana poly failed to find all the surface : just sees a small portion of it
     op --dlv185 --gltf 3  # viewing ana raytrace together with tri poly  : looks OK, open tube on top of SST 

     op --dlv185 --gmeshlib --dbgmesh MOFTTube0xc046b40



DBGMESH=MOFTTube0xc046b40 NSceneMeshTest
-------------------------------------------

::

    simon:issues blyth$ DBGMESH=MOFTTube0xc046b40 NSceneMeshTest
    2017-07-04 13:51:22.707 INFO  [3105477] [main@29] NSceneMeshTest gltfbase /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300 gltfname g4_00.gltf gltfconfig check_surf_containment=0,check_aabb_containment=0
    2017-07-04 13:51:22.708 INFO  [3105477] [NGLTF::load@35] NGLTF::load path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
    ...
    2017-07-04 13:51:27.123 INFO  [3105477] [NScene::init@202] NScene::init DONE
    2017-07-04 13:51:27.123 INFO  [3105477] [NScene::dumpCSG@434] NScene::dumpCSG num_csg 249 dbgmesh MOFTTube0xc046b40
    2017-07-04 13:51:27.124 INFO  [3105477] [NCSG::dump@907] NCSG::dump
     NCSG  ix  195 surfpoints   55 so MOFTTube0xc046b40                        lv /dd/Geometry/AdDetails/lvMOFTTube0xbfa58b0
    NCSG::dump (root) [ 0:di] OPER  v:0
             L [ 1:cy] PRIM  v:0 bb  mi  (-581.00 -581.00 -127.50)  mx  ( 581.00  581.00  127.50)  si  (1162.00 1162.00  255.00) 
             R [ 2:cy] PRIM  v:0 bb  mi  (-575.00 -575.00 -128.77)  mx  ( 575.00  575.00  128.77)  si  (1150.00 1150.00  257.55) 
     composite_bb  mi  (-581.00 -581.00 -127.50)  mx  ( 581.00  581.00  127.50)  si  (1162.00 1162.00  255.00) 
    NParameters::dump
             lvname : /dd/Geometry/AdDetails/lvMOFTTube0xbfa58b0
             soname : MOFTTube0xc046b40
          verbosity :               0
         resolution :              20
               poly :              IM
             height :               1
    2017-07-04 13:51:27.124 INFO  [3105477] [NCSG::dump_surface_points@1197] dsp num_sp 55 dmax 200
     i    0 sp (    581.000     0.000  -127.500)
     i    1 sp (     -0.000   581.000  -127.500)
     i    2 sp (   -581.000    -0.000  -127.500)
     i    3 sp (      0.000  -581.000  -127.500)
     i    4 sp (    581.000     0.000  -127.500)
     i    5 sp (    581.000     0.000   -66.938)
     i    6 sp (     -0.000   581.000   -66.938)
     i    7 sp (   -581.000    -0.000   -66.938)
     ...
     i   52 sp (   -575.000    -0.000   115.897)
     i   53 sp (      0.000  -575.000   115.897)
     i   54 sp (    575.000     0.000   115.897)
     csg.index (mesh_id) 195 num nodes 4
     node idx :  4799 4807 6459 6467 . 



