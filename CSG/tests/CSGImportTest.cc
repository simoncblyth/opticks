/**
CSGImportTest.cc
=================

1. loads stree from file
2. populates CSGFoundry with CSGFoundry::importTree 

**/

#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "stree.h"

#include "CSGFoundry.h"

const char* BASE = getenv("BASE");  
const char* FOLD = getenv("FOLD");  

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SSim* sim = SSim::Create(); 

    stree* st = sim->tree ;  
    st->level = 2 ; 
    st->load(BASE) ;  
    std::cout << st->desc() ; 

    CSGFoundry* fd = new CSGFoundry ; 
    fd->importTree( st ); 
    if(FOLD) fd->save(FOLD) ; 

    return 0 ;  
}



