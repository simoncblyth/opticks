#include <cassert>

#include "SAr.hh"
#include "SSys.hh"

int main(int argc, char** argv)
{
    SAr a(argc, argv );
    a.dump();
    a.dump();

    const char* key = "SAR_TEST" ; 
    const char* val = "--trace --SYSRAP warning red green blue" ; 
    bool overwrite = true ; 
    SSys::setenvvar( key, val , overwrite ) ;  

    SAr b(0,0, key, ' ') ;
    b.dump() ; 

    assert( b._argc == 7 ); 


    return 0 ; 
}
