
#include "PLOG.hh"

#include "SSys.hh"
#include "Opticks.hh"
#include "CG4.hh"

#include "Randomize.hh"
#include "CLHEP/Random/NonRandomEngine.h"

#include "G4String.hh"
#include "G4VProcess.hh"

#include "CProcess.hh"
#include "CRandomEngine.hh"






std::string CRandomEngine::name() const 
{
    return "CRandomEngine";
}

CRandomEngine::CRandomEngine(CG4* g4)
    :
    m_g4(g4),
    m_ctx(g4->getCtx()),
    m_ok(g4->getOpticks()),
    m_seed(9876),
    m_james(new CLHEP::HepJamesRandom()),
    m_nonran(new CLHEP::NonRandomEngine()),
    m_engine(m_james),
    m_count(0),
    m_harikari(SSys::getenvint("OPTICKS_CRANDOMENGINE_HARIKARI", -1 ))
{
    init();
}

void CRandomEngine::init()
{
    //m_nonran->setRandomSequence( seq.data(), seq.size() ) ; 

    CLHEP::HepRandom::setTheEngine( this );  
    CLHEP::HepRandom::setTheSeed(  m_seed );    // does nothing for NonRandom

    // want flat calls to go via this instance, so can check on em 
}



bool CRandomEngine::isNonRan() const 
{
    return m_engine == m_nonran ; 
}
bool CRandomEngine::isDefault() const 
{
    return m_engine == m_james ; 
}


double CRandomEngine::flat() 
{ 
    double _flat =  m_engine->flat() ;  

    G4VProcess* proc = CProcess::CurrentProcess() ; 
    const G4String& name = proc->GetProcessName()  ; 

    LOG(info) << " record_id " << m_ctx._record_id 
              << " count " << m_count
              << " flat " << _flat 
              << " processName " << name
              ; 

    m_record_count[m_ctx._record_id]++ ; 

    if( int(m_count) == m_harikari ) 
    {
        LOG(error) << "OPTICKS_CRANDOMENGINE_HARIKARI" ; 
        assert(0) ;  
    }

    m_count++ ; 

    return _flat ; 
}




void CRandomEngine::dump(const char* msg) const 
{
    LOG(info) << msg ; 

    typedef std::map<unsigned, unsigned> UMM ; 
    for(UMM::const_iterator it=m_record_count.begin() ; it != m_record_count.end() ; it++ )
    {
        std::cout 
            << std::setw(10) << it->first 
            << std::setw(10) << it->second
            << std::endl 
            ;
    }
}

void CRandomEngine::postpropagate()
{
    dump("CRandomEngine::postpropagate");
}



void CRandomEngine::flatArray(const int size, double* vect) 
{
    for (int i = 0; i < size; ++i) 
    {
        vect[i] = flat();
    }
}


void CRandomEngine::setSeed(long seed, int dum) 
{
    if(isNonRan()) 
    {
        LOG(info) << "CRandomEngine::setSeed ignoring " ; 
        return ;
    }

    m_engine->setSeed(seed, dum);
} 


void CRandomEngine::setSeeds(const long * seeds, int dum) 
{
    if(isNonRan()) 
    {
        LOG(info) << "CRandomEngine::setSeeds ignoring " ; 
        return ;
    }

    m_engine->setSeeds(seeds, dum);
}

void CRandomEngine::saveStatus( const char * filename ) const 
{
    if(isNonRan()) 
    {
        LOG(info) << "CRandomEngine::saveStatus ignoring " ; 
        return ;
    }
    m_engine->saveStatus(filename) ; 

}
        
void CRandomEngine::restoreStatus( const char * filename)
{
    if(isNonRan()) 
    {
        LOG(info) << "CRandomEngine::restoreStatus ignoring " ; 
        return ;
    }
    m_engine->restoreStatus(filename) ; 
}


void CRandomEngine::showStatus() const 
{
    if(isNonRan()) 
    {
        LOG(info) << "CRandomEngine::showStatus ignoring " ; 
        return ;
    }
    m_engine->showStatus() ; 
}



