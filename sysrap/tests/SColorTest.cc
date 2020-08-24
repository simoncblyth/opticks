#include "SColor.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    assert( SColors::red.r == 0xff ); 

    assert( SColors::red.get(0) == 0xff ); 
    assert( SColors::red.get(1) == 0x00 ); 
    assert( SColors::red.get(2) == 0x00 ); 
    assert( SColors::red.get(3) == 0xff ); 


    return 0 ; 
}
