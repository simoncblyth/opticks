// op --cproplib 
// op --cproplib 0
// op --cproplib GdDopedLS

#include "Opticks.hh"
#include "OpticksMode.hh"
#include "CPropLib.hh"
#include "CFG4_BODY.hh"
#include "CFG4_LOG.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    CFG4_LOG__ ;  

    LOG(info) << argv[0] ; 

    Opticks* m_opticks = new Opticks(argc, argv);
    
    m_opticks->setModeOverride( OpticksMode::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    CPropLib* m_lib = new CPropLib(m_opticks); 

    LOG(info) << argv[0] << " convert " ; 

    m_lib->convert();

    LOG(info) << argv[0] << " dump " ; 

    m_lib->dump();

    //m_lib->dumpMaterials();



    return 0 ; 
}
