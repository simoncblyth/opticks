Polygonization Sqrt3 Subdiv
=============================

Overview
---------

* progress kinda stymied by subdiv problems, 

  perhaps OpenMesh multi-flip bug ...
  so try out OpenFlipper which is using newer OpenMesh


* actually polygonization is a once only task for a geometry, so 
  taking a higher level view and using existing tools is entirely 
  plausible : and of course if can get it to work at 
  a high level can batch it somehow
  

Higher Level Approaches
-------------------------

* https://github.com/VTREEM/Carve





BASH Funcs
-------------



::


     807 tboolean-hyctrl(){ TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
     808 #tboolean-hyctrl-polytest(){ lldb NPolygonizerTest -- $TMP/tboolean-hyctrl--/1 ; }
     809 tboolean-hyctrl-polytest(){ NPolygonizerTest $TMP/tboolean-hyctrl--/1 ; }
     810 tboolean-hyctrl-(){ $FUNCNAME- | python $* ; }
     811 tboolean-hyctrl--(){ cat << EOP
     812 from opticks.analytic.csg import CSG  
     813 
     814 container = CSG("box",   name="container",  param=[0,0,0,1000], boundary="$(tboolean-container)", poly="IM", resolution="1" )
     815 
     816 
     817 #ctrl = "1"  # subdiv_test
     818 #ctrl = "4"  # tetrahedron
     819 #ctrl = "6"  # cube
     820 ctrl = "66" # hexpatch inner_only 
     821 #ctrl = "666" # hexpatch
     822 
     823 box = CSG("box", param=[0,0,0,500], boundary="$(tboolean-testobject)", poly="HY", level="0", ctrl=ctrl, verbosity="3" )
     824 
     825 CSG.Serialize([container, box ], "$TMP/$FUNCNAME" )
     826 
     827 EOP
     828 }


::


    delta:opticksnpy blyth$ tboolean-;tboolean-hyctrl-polytest
    2017-06-05 11:46:57.064 INFO  [5038581] [main@17]  argc 2 argv[0] NPolygonizerTest
    2017-06-05 11:46:57.065 INFO  [5038581] [NCSG::import@442] NCSG::import START verbosity 3 treedir /tmp/blyth/opticks/tboolean-hyctrl--/1 smry  ht  0 nn    1 tri      0 tmsg NULL-tris iug 1 nd 1,4,4 tr 1,3,4,4 gtr 0,3,4,4 pln NULL
    2017-06-05 11:46:57.065 INFO  [5038581] [NCSG::import@450] NCSG::import importing buffer into CSG node tree  num_nodes 1 height 0
    2017-06-05 11:46:57.065 INFO  [5038581] [NCSG::import_primitive@587] NCSG::import_primitive   idx 0 typecode 6 csgname box
    2017-06-05 11:46:57.065 INFO  [5038581] [NCSG::import@462] NCSG::import DONE 
    NPolygonizer::NPolygonizer(meta)
          verbosity :               3
               ctrl :              66
               poly :              HY
              level :               0
    2017-06-05 11:46:57.065 INFO  [5038581] [NPolygonizer::polygonize@55] NPolygonizer::polygonize treedir /tmp/blyth/opticks/tboolean-hyctrl--/1 poly HY verbosity 3 index 0
    2017-06-05 11:46:57.065 INFO  [5038581] [NHybridMesher::make_mesh@17] NHybridMesher::make_mesh level 0 verbosity 3 ctrl 66
    2017-06-05 11:46:57.065 INFO  [5038581] [>::add_hexpatch@525] add_hexpatch
    2017-06-05 11:46:57.066 INFO  [5038581] [>::check@59] NOpenMesh<T>::check OK
    2017-06-05 11:46:57.066 INFO  [5038581] [>::find_boundary_loops@128] find_boundary_loops
    NOpenMeshBoundary start 3 collected 6 :  3 23 19 15 11 7...
    2017-06-05 11:46:57.066 INFO  [5038581] [>::find_boundary_loops@189] find_boundary_loops he_bnd[0] 0 he_bnd[1] 1 he_bnd[2] 12 loops 1
    2017-06-05 11:46:57.066 INFO  [5038581] [>::find_faces@53] NOpenMeshFind<T>::find_faces (faces with all vertices having same valence)  select 0 param 6 count 6 totface 6
    2017-06-05 11:46:57.066 INFO  [5038581] [>::subdiv_test@82] subdiv_test ctrl 66 verbosity 3 n_target_faces 6 nloop0 1
    sqrt3_split_r face fh    0 base_gen    0 base_id    1 depth    0
     (even)  newface_id  101 newface_gen    1 adjacent_valid Y adjacent_id    2 adjacent_gen    0 do_flip NO
     (even)  newface_id  102 newface_gen    1 adjacent_valid N
     (even)  newface_id  103 newface_gen    1 adjacent_valid Y adjacent_id    6 adjacent_gen    0 do_flip NO
    sqrt3_split_r face fh    1 base_gen    0 base_id    2 depth    0
     (even)  newface_id  201 newface_gen    1 adjacent_valid Y adjacent_id    3 adjacent_gen    0 do_flip NO
     (even)  newface_id  202 newface_gen    1 adjacent_valid N
     (even)  newface_id  203 newface_gen    1 adjacent_valid Y adjacent_id  101 adjacent_gen    1 do_flip YES
    sqrt3_split_r face fh    2 base_gen    0 base_id    3 depth    0
     (even)  newface_id  301 newface_gen    1 adjacent_valid Y adjacent_id    4 adjacent_gen    0 do_flip NO
     (even)  newface_id  302 newface_gen    1 adjacent_valid N
     (even)  newface_id  303 newface_gen    1 adjacent_valid Y adjacent_id  201 adjacent_gen    1 do_flip YES
    sqrt3_split_r face fh    3 base_gen    0 base_id    4 depth    0
     (even)  newface_id  401 newface_gen    1 adjacent_valid Y adjacent_id    5 adjacent_gen    0 do_flip NO
     (even)  newface_id  402 newface_gen    1 adjacent_valid N
     (even)  newface_id  403 newface_gen    1 adjacent_valid Y adjacent_id  301 adjacent_gen    1 do_flip YES
    sqrt3_split_r face fh    4 base_gen    0 base_id    5 depth    0
     (even)  newface_id  501 newface_gen    1 adjacent_valid Y adjacent_id    6 adjacent_gen    0 do_flip NO
     (even)  newface_id  502 newface_gen    1 adjacent_valid N
     (even)  newface_id  503 newface_gen    1 adjacent_valid Y adjacent_id  401 adjacent_gen    1 do_flip YES
    sqrt3_split_r face fh    5 base_gen    0 base_id    6 depth    0
     (even)  newface_id  601 newface_gen    1 adjacent_valid Y adjacent_id  103 adjacent_gen    1 do_flip YES
     (even)  newface_id  602 newface_gen    1 adjacent_valid N
     (even)  newface_id  603 newface_gen    1 adjacent_valid Y adjacent_id  501 adjacent_gen    1 do_flip YES

     (even)  newface_id  604 newface_gen    1 adjacent_valid Y adjacent_id  203 adjacent_gen    2 do_flip NO       
           ^^^^^^^  203 is post-flip ???  ^^^^^^^^

    2017-06-05 11:46:57.066 INFO  [5038581] [>::find_boundary_loops@128] find_boundary_loops
    NOpenMeshBoundary start 3 collected 6 :  3 23 19 15 11 7...
    2017-06-05 11:46:57.066 INFO  [5038581] [>::find_boundary_loops@189] find_boundary_loops he_bnd[0] 0 he_bnd[1] 1 he_bnd[2] 30 loops 1
    2017-06-05 11:46:57.066 INFO  [5038581] [>::subdiv_test@101] subdiv_test DONE  ctrl 66 verbosity 3 nloop1 1
    2017-06-05 11:46:57.066 INFO  [5038581] [*NPolygonizer::polygonize@99] NPolygonizer::polygonize OK  numTris 18
    delta:opticksnpy blyth$ 
    delta:opticksnpy blyth$ 

::


       //        base_id:1 adjacent:   2,   -,   6    
       //                            101, 102, 103 
       //
       //        base_id:2 adjacent:   3,   -, 101
       //                            201, 202, 203*
       //     
       //        base_id:3 adjacent:   4,   -, 201
       //                            301, 302, 303*
       //
       //        base_id:4 adjacent:   5,   -, 301
       //                            401, 402, 403*
       //  
       //        base_id:5 adjacent:   6,   -, 401
       //                            501, 502, 503*
       //  
       //        base_id:6 adjacent:  103,   -, 501
       //                             601, 602, 603
       //  


                                                                     
                 +-------+        
                / \     / \      
               /   \ 2 /   \      
              /  3  \ /  1  \     
             +-------+-------+   
              \  4  / \  6  /   
               \   / 5 \   /   
                \ /     \ /   
                 +-------+   


