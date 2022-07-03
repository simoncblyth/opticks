
#include <iostream>
#include <cstdlib>

#include "SPath.hh"
#include "SDir.h"
#include "OPTICKS_LOG.hh"


void test_List_npy()
{
    std::vector<std::string> names ; 
    const char* dir = SPath::Resolve("$PrecookedDir/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000", DIRPATH ); 

    LOG(info) << dir ; 

    SDir::List(names, dir,  ".npy" );
    std::cout << SDir::Desc(names) << std::endl ;
}

void test_List_ori()
{
    const char* mlib = SPath::Resolve("$IDPath/GMaterialLib", DIRPATH); 
    LOG(info) << mlib ;  

    std::vector<std::string> names ; 
    SDir::List(names, mlib, "_ori" );
    SDir::Trim(names, "_ori" );  
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
