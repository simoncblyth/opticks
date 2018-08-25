#include <cstring>
#include <cassert>

#include "CFG4_BODY.hh"
#include "CPrimaryCollector.hh"
#include "CSource.hh"

CSource::CSource(Opticks* ok )  
    :
    m_ok(ok),
    m_recorder(NULL)
{
}

CSource::~CSource()
{
}  

void CSource::setRecorder(CRecorder* recorder)
{
    m_recorder = recorder ;  
}


NPY<float>* CSource::getSourcePhotons() const
{
    return NULL ; 
}

void CSource::collectPrimaryVertex(const G4PrimaryVertex* vtx)
{
    CPrimaryCollector* pc = CPrimaryCollector::Instance() ;
    assert( pc ); 
    G4int vertex_index = 0 ;    // assumption 
    pc->collectPrimaryVertex(vtx, vertex_index); 
}



