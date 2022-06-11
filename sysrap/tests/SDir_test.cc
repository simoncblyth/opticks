// export OPTICKS_RANDOM_SEQPATH=/tmp/$USER/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000
// name=SDir_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include <cstdlib>
#include "SDir.h"


void test_List_npy()
{
    std::vector<std::string> names ; 
    SDir::List(names, getenv("OPTICKS_RANDOM_SEQPATH"), ".npy" );
    std::cout << SDir::Desc(names) << std::endl ;
}

void test_List_ori()
{
    std::stringstream ss ;
    ss << getenv("IDPath") << "/GMaterialLib" ; 
    std::string mlib = ss.str(); 

    std::vector<std::string> names ; 
    SDir::List(names, mlib.c_str(), "_ori" );
    SDir::Trim(names, "_ori" );  
    std::cout << SDir::Desc(names) << std::endl ;
}


int main()
{
    /*
    test_List_npy(); 
    */
    test_List_ori(); 

    return 0 ; 
}
