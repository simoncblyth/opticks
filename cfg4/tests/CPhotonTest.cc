
#include "PLOG.hh"
#include "CPhoton.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    CPhoton p ; 

    for(unsigned i=0 ; i < 15 ; i++)
    {
        unsigned slot = i ; 
        unsigned flag = 0x1 << i ; 
        unsigned material = i + 1 ; 
        p.add( slot, flag, material );
    }

    LOG(info) << p.desc() ; 


    return 0 ; 
}
 
