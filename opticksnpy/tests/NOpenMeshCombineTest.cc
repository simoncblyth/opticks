/**
Tests boolean combination of parametric meshes::

    NOpenMeshCombineTest $TMP/tboolean-hybrid--/1


::

    simon:opticks blyth$ lldb NOpenMeshCombineTest
    (lldb) target create "NOpenMeshCombineTest"
    Current executable set to 'NOpenMeshCombineTest' (x86_64).
    (lldb) r
    Process 44222 launched: '/usr/local/opticks/lib/NOpenMeshCombineTest' (x86_64)
    2017-05-31 18:20:21.109 INFO  [4183530] [main@31]  argc 1 argv[0] NOpenMeshCombineTest
    2017-05-31 18:20:21.111 INFO  [4183530] [>::build_parametric_primitive@754] NOpenMesh<T>::build_parametric ns 6 nu 16 nv 16 num_vert(raw) 1734 epsilon 1e-05
    2017-05-31 18:20:21.238 INFO  [4183530] [>::build_parametric_primitive@941]  V 1538 E 4608 F 3072 Euler [(V - E + F)] 2
    2017-05-31 18:20:21.238 INFO  [4183530] [>::build_parametric_primitive@942] build_parametric euler 2 expect_euler 2 EULER_OK  nvertices 1538 expect_nvertices 1538 NVERTICES_OK 
    2017-05-31 18:20:21.238 INFO  [4183530] [>::build_parametric_primitive@754] NOpenMesh<T>::build_parametric ns 1 nu 16 nv 16 num_vert(raw) 289 epsilon 1e-05
    2017-05-31 18:20:21.242 INFO  [4183530] [>::build_parametric_primitive@941]  V 242 E 720 F 480 Euler [(V - E + F)] 2
    2017-05-31 18:20:21.242 INFO  [4183530] [>::build_parametric_primitive@942] build_parametric euler 2 expect_euler 2 EULER_OK  nvertices 242 expect_nvertices 242 NVERTICES_OK 
    2017-05-31 18:20:21.242 INFO  [4183530] [>::build_parametric@96] build_parametric leftmesh 0x102c05cd0 rightmesh 0x102c06600
    2017-05-31 18:20:21.247 INFO  [4183530] [>::build_parametric@102] leftmesh inside node->right :   0 :   2728|  1 :     14|  2 :     14|  3 :     12|  4 :     28|  5 :     12|  6 :     12|  7 :    252|
    2017-05-31 18:20:21.247 INFO  [4183530] [>::build_parametric@105] rightmesh inside node->left :   0 :     72|  1 :     10|  2 :     10|  3 :     36|  4 :      8|  5 :     16|  6 :     16|  7 :    312|
    2017-05-31 18:20:21.803 INFO  [4183530] [>::dump_boundary_faces@167] boundary faces
    facemask:4
     quv:   4  1 16 16 pt:         200.000       -100.000       -175.000 _a_sdf:           0.000 _b_sdf:          25.000
     quv:   5  1 16 16 pt:         200.000        -75.000       -175.000 _a_sdf:           0.000 _b_sdf:          15.058
     quv:   5  2 16 16 pt:         200.000        -75.000       -150.000 _a_sdf:           0.000 _b_sdf:          -4.744
    facemask:1
     quv:   5  2 16 16 pt:         200.000        -75.000       -150.000 _a_sdf:           0.000 _b_sdf:          -4.744
     quv:   4  2 16 16 pt:         200.000       -100.000       -150.000 _a_sdf:           0.000 _b_sdf:           6.155
     quv:   4  1 16 16 pt:         200.000       -100.000       -175.000 _a_sdf:           0.000 _b_sdf:          25.000
    facemask:4
     quv:   5  1 16 16 pt:         200.000        -75.000       -175.000 _a_sdf:           0.000 _b_sdf:          15.058
     quv:   6  1 16 16 pt:         200.000        -50.000       -175.000 _a_sdf:           0.000 _b_sdf:           7.666
     quv:   5  2 16 16 pt:         200.000        -75.000       -150.000 _a_sdf:           0.000 _b_sdf:          -4.744
    facemask:3
     quv:   6  2 16 16 pt:         200.000        -50.000       -150.000 _a_sdf:           0.000 _b_sdf:         -12.917
     quv:   5  2 16 16 pt:         200.000        -75.000       -150.000 _a_sdf:           0.000 _b_sdf:          -4.744
     quv:   6  1 16 16 pt:         200.000        -50.000       -175.000 _a_sdf:           0.000 _b_sdf:           7.666
    facemask:4



**/

#include <iostream>

#include "BStr.hh"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NOpenMesh.hpp"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

 


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    const char* treedir = argc > 1 ? argv[1] : "$TMP/tboolean-hybrid--/1" ;

    int verbosity = 2 ; 

    NCSG* tree = NCSG::LoadTree(treedir, verbosity );

    assert( tree );

    const nnode* root = tree->getRoot();

    int level = 4 ; 
    int ctrl = 0 ; 

    NOpenMesh<NOpenMeshType> mesh(root, level, verbosity, ctrl );



    return 0 ; 
}


