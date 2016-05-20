#include "CDetector.hh"

// npy-
#include "NLog.hpp"
#include "GLMFormat.hpp"

// cfg4-
#include "CPropLib.hh"

// ggeo-
#include "GCache.hh"


// g4-
//#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"


void CDetector::init()
{
    m_lib = new CPropLib(m_cache);
}


void CDetector::dumpPV(const char* msg)
{
    LOG(info) << msg ; 

    typedef std::map<std::string, G4VPhysicalVolume*> MSV ; 

    for(MSV::const_iterator it=m_pvm.begin() ; it != m_pvm.end() ; it++)
    {
         std::string pvn = it->first ; 
         G4VPhysicalVolume* pv = it->second ;  

         std::cout << std::setw(40) << pvn 
                   << std::setw(40) << pv->GetName() 
                   << std::endl 
                   ;

    }
}

G4VPhysicalVolume* CDetector::getPV(const char* name)
{
    return m_pvm.count(name) == 1 ? m_pvm[name] : NULL ; 
}

