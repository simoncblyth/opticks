/**
SRecorder_test.cc
================

CreateFromTree
---------------

1. Load stree.h from TREE_FOLD
2. SRecorder::initFromTree 
3. saves scene to SCENE_FOLD


Load
-----

1. SRecorder::Load from SCENE_FOLD
2. SRecorder::desc to stdout 


Usage
-------

::

    TEST=CreateFromTree ~/o/sysrap/tests/SRecorder_test.sh 
    TEST=Load ~/o/sysrap/tests/SRecorder_test.sh 
   ~/o/sysrap/tests/SRecorder_test.sh 

   ~/o/sysrap/tests/SRecorder_test.cc

**/

#include "ssys.h"
#include "SRecorder.h"

struct SRecorder_test
{
    static int Load(); 
    static int Main();
}; 


inline int SRecorder_test::Load()
{
    std::cout << "[SRecorder_test::Load" << std::endl ; 
    SRecorder* sr= SRecorder::Load("$SRECORD_PATH") ; 
    sr->init_minmax2D();
    sr->desc() ; 
    std::cout << "]SRecorder_test::Load" << std::endl ; 
    return 0 ; 
}

inline int SRecorder_test::Main()
{
    int rc(0) ; 
    const char* TEST = ssys::getenvvar("TEST", "Load"); 
  
    if ( strcmp(TEST,"Load") == 0 )           rc += Load() ;   

    return rc ;  
}

int main()
{
    return SRecorder_test::Main();
}

