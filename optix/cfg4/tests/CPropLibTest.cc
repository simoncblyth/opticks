// ggv --cproplib 

#include "Opticks.hh"
#include "GCache.hh"
#include "CPropLib.hh"


#include "NLog.hpp"

int main(int argc, char** argv)
{
    LOG(info) << "klop " ;

    Opticks* m_opticks = new Opticks(argc, argv, "CPropLibTest.log");
    
    m_opticks->setMode( Opticks::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    GCache* m_cache = new GCache(m_opticks);

    CPropLib* m_lib = new CPropLib(m_cache); 

    m_lib->dumpMaterials();


    return 0 ; 
}
