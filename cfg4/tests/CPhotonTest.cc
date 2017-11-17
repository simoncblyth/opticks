
#include <vector>

#include "PLOG.hh"
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"
#include "CG4Ctx.hh"
#include "CRecState.hh"
#include "CPhoton.hh"



void test_CPhoton(CPhoton& p, const std::vector<unsigned>& flags )
{
    for(unsigned i=0 ; i < flags.size() ; i++)
    {
        unsigned flag = flags[i];
        unsigned material = i + 1 ; // placeholder
 
        p.add( flag, material );
        p.increment_slot();

        LOG(info) << p.desc() ; 
    }
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    Opticks ok(argc, argv );
    ok.configure();
    ok.setSpaceDomain(0,0,0,0); // for configureDomains

    OpticksEvent* evt = ok.makeEvent(false, 0u);

    CG4Ctx ctx(&ok);
    ctx.initEvent(evt);

    CRecState s(ctx);
    CPhoton   p(ctx, s) ; 

    {
        std::vector<unsigned> flags ; 
        flags.push_back(TORCH);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_REFLECT);
        flags.push_back(BULK_ABSORB);

        p.clear();
        s.clear();

        test_CPhoton(p, flags );
    }

    {
        std::vector<unsigned> flags ; 
        flags.push_back(TORCH);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_REFLECT);
        flags.push_back(BULK_SCATTER);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(SURFACE_DREFLECT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(BOUNDARY_TRANSMIT);
        flags.push_back(SURFACE_SREFLECT);
        flags.push_back(BULK_ABSORB);


        p.clear();
        s.clear();

        test_CPhoton(p, flags );
    }


    return 0 ; 
}
 
