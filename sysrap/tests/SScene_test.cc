/**
SScene_test.cc
================

::

   ~/o/sysrap/tests/SScene_test.sh 

**/

#include "SScene.h"

int main()
{
    stree* st = stree::Load("$STREE_FOLD"); 
    std::cout << st->desc() ; 

    SScene scene ; 
    scene.initFromTree(st); 

    std::cout << scene.desc() ; 
    scene.save("$SSCENE_FOLD/scene/mesh") ;  

    return 0 ; 
}
