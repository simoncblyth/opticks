#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksMode.hh"

#include "CMaterialLib.hh"
#include "CMPT.hh"
#include "CVec.hh"

#include "CFG4_BODY.hh"
#include "OPTICKS_LOG.hh"

#include "PLOG.hh"


void test_CMPT(CMaterialLib* clib)
{
    const char* shortname = "Acrylic" ; 
    const CMPT* mpt = clib->getG4MPT(shortname);
    mpt->dump(shortname);

    const char* lkey = "GROUPVEL" ; 

    CVec* vec = mpt->getCVec(lkey);
    vec->dump(lkey);
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    Opticks ok(argc, argv);
    
    ok.setModeOverride( OpticksMode::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    OpticksHub hub(&ok); 

    CMaterialLib* clib = new CMaterialLib(&hub); 

    LOG(info) << argv[0] << " convert " ; 

    clib->convert();

    LOG(info) << argv[0] << " dump " ; 

    //clib->dump();



    const char* shortname = "Pyrex" ;

    const CMPT* mpt = clib->getG4MPT(shortname);
    mpt->dump(shortname);

    const char* lkey = "ABSLENGTH" ; 

    CVec* vec = mpt->getCVec(lkey);
    vec->dump(lkey);

    return 0 ; 
}



