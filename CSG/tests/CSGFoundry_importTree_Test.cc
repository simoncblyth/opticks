/**
CSGMaker_importTree_Test.cc
=============================

**/


#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "stree.h"

#include "CSGFoundry.h"

const char* BASE = getenv("BASE");  

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


    return 0 ;  
}



