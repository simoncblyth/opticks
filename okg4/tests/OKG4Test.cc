
#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksHub.hh"

#include "CG4.hh"
#include "CMaterialTable.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
  
    Opticks* m_opticks = new Opticks(argc, argv);
    m_opticks->setModeOverride( OpticksMode::CFG4_MODE );

    OpticksHub* m_hub = new OpticksHub(m_opticks) ; 
    m_hub->configure();
    m_hub->loadGeometry();


    CG4* g4 = new CG4(m_hub) ; 
    g4->configure();
    g4->initialize();

    CMaterialTable* m_mtab = new CMaterialTable(m_opticks->getMaterialPrefix());
    m_mtab->dump("OKG4Test CMaterialTable");


/*
    // where should lookup live ? GBndLib seems most appropriate 
    NLookup* m_lookup = new NLookup();   

    std::map<std::string, unsigned>& A = m_lookup->getA() ;
    std::map<std::string, unsigned>& B = m_lookup->getB() ;

    m_mtab->fillMaterialIndexMap( A ) ; 

*/



    return 0 ;
}
