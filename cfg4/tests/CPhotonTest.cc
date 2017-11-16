
#include "PLOG.hh"
#include "Opticks.hh"
#include "CG4Ctx.hh"
#include "CPhoton.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    Opticks ok(argc, argv );
    ok.configure();

    CG4Ctx ctx(&ok);
    CPhoton p(ctx) ; 

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
 
