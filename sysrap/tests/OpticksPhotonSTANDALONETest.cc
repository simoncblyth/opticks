// ~/opticks/sysrap/tests/OpticksPhotonSTANDALONETest.sh

#include <iostream>
#include <iomanip>

#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"

#include "spath.h"
#include "NP.hh"

void test_list()
{
    for(unsigned i=0 ; i < 16 ; i++) 
    {
        unsigned flag = 0x1 << i ; 
        std::cout 
            << " i " << std::setw(3) << i 
            << " flag " << std::setw(10) << flag 
            << " OpticksPhoton::Flag " << std::setw(20)  << OpticksPhoton::Flag(flag) 
            << " OpticksPhoton::Abbrev " << std::setw(4) << OpticksPhoton::Abbrev(flag) 
            << std::endl 
            ;
    }
}



void test_load_seq()
{
    const char* _path = "$TMP/GEOM/$GEOM/G4CXTest/ALL0/p001/seq.npy" ; 
    const char* path = spath::Resolve(_path) ; 
    NP* a = NP::Load(path); 
    std::cout 
        << "OpticksPhotonTest:test_load_seq" 
        << std::endl
        << " _path " << _path 
        << std::endl
        << " path  " << path 
        << std::endl
        << " a " << ( a ? a->sstr() : "-" ) 
        << std::endl
         ; 
    if(a == nullptr) return ; 

    const uint64_t* aa = a->cvalues<uint64_t>(); 
    int num = a->shape[0] ;  
    int edge = 10 ; 

    for(int i=0 ; i < num ; i++)
    {
        if( i < edge || i > (num - edge) ) 
            std::cout << OpticksPhoton::FlagSequence_( aa + 4*i, 2 ) << std::endl ;  
        else if( i == edge ) 
            std::cout << "..." << std::endl ;
    }
}


int main(int argc, char** argv)
{
    test_load_seq(); 
    return 0 ; 
}
