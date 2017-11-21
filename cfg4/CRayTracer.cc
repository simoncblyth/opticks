
#include "G4TheRayTracer.hh"


#include "BFile.hh"
#include "OpticksHub.hh"

#include "CG4.hh"
#include "CRayTracer.hh"

#include "PLOG.hh"

CRayTracer::CRayTracer(CG4* g4)
    :
    m_g4(g4),
    m_ok(g4->getOpticks()),
    m_hub(g4->getHub()),
    m_composition(m_hub->getComposition()),

    m_figmaker(NULL),
    m_scanner(NULL),
    m_tracer( new G4TheRayTracer(m_figmaker, m_scanner) )
{
}


void CRayTracer::snap() const 
{
    std::string path_ = BFile::FormPath("$TMP/CRayTracer.jpeg"); 

    LOG(info) << "path " << path_ ; 
 
    G4String path = path_ ; 

    m_tracer->Trace( path );
  
}


