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

    SScene sc(st) ; 
    std::cout << "sc.desc" << sc.desc() ; 

    return 0 ; 
}
