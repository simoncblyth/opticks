
#include <iostream>
#include <cstdlib>

#include "spath.h"
#include "SDir.h"
#include "OPTICKS_LOG.hh"


void test_List_npy()
{
    std::vector<std::string> names ; 
    const char* dir = spath::Resolve("$HOME/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"); 

    LOG(info) << dir ; 

    SDir::List(names, dir,  ".npy" );
    std::cout << SDir::Desc(names) << std::endl ;
}

void test_List_ori()
{
    const char* mlib = spath::Resolve("$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/material"); 
    LOG(info) << mlib ;  

    std::vector<std::string> names ; 
    SDir::List(names, mlib, "" );
    std::cout << SDir::Desc(names) << std::endl ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    /*
    */
    test_List_npy(); 
    test_List_ori(); 

    return 0 ; 
}
