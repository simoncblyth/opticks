
#include "NPY.hpp"
#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksMode.hh"
#include "CMaterialLib.hh"
#include "CFG4_BODY.hh"
#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_COLOR(argc, argv);

    GGEO_LOG__ ;  
    CFG4_LOG__ ;  

    LOG(info) << argv[0] ; 

    Opticks ok(argc, argv);
    
    ok.setModeOverride( OpticksMode::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    OpticksHub hub(&ok); 

    CMaterialLib* clib = new CMaterialLib(&hub); 

    LOG(info) << argv[0] << " convert " ; 

    clib->convert();

    LOG(info) << argv[0] << " dump " ; 

    clib->dump();

    clib->saveGROUPVEL("$TMP/CGROUPVELTest");



    return 0 ; 
}
