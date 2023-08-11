/**
CSGImportTest.cc
=================

1. loads stree from file
2. populates CSGFoundry with CSGFoundry::importTree 

**/

#include "OPTICKS_LOG.hh"
#include "stree.h"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    stree* st = stree::Load("$BASE") ; 
    st->level = 2 ; 
    std::cout << st->desc() ; 

    CSGFoundry* fd = CSGFoundry::Import(st) ; 
    fd->save("$FOLD") ; 

    return 0 ;  
}



