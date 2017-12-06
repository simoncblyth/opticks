
#include <array>

#include "PLOG.hh"

#include "Randomize.hh"
#include "CLHEP/Random/NonRandomEngine.h"
#include "G4String.hh"
#include "G4VProcess.hh"


#include "SSys.hh"

#include "BStr.hh"
#include "BFile.hh"
#include "BLocSeq.hh"

#include "Opticks.hh"

#include "CG4.hh"
#include "CProcess.hh"
#include "CStepStatus.hh"
#include "CStepping.hh"
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
    m_internal(false),
    m_skipdupe(true),
    m_locseq(new BLocSeq<unsigned long long>(m_skipdupe)),
    m_james(new CLHEP::HepJamesRandom()),
    m_nonran(new CLHEP::NonRandomEngine()),
    m_engine(m_james)
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

std::string CRandomEngine::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " record_id " << std::setw(5) << m_ctx._record_id 
       << " step_id " << std::setw(5) << m_ctx._step_id
       << " loc " << std::setw(50) << m_location 
       ;

    return ss.str();
}


std::string CRandomEngine::FormLocation()
{
    G4VProcess* proc = CProcess::CurrentProcess() ; 
    const G4String& procName = proc->GetProcessName()  ; 
    std::stringstream ss ; 
    ss << procName << ";" ;  
    return ss.str();
}

std::string CRandomEngine::FormLocation(const char* file, int line)
{
    assert( file ) ;
    std::stringstream ss ; 
    std::string relpath = BFile::prefixShorten(file, "$OPTICKS_HOME/" );
    ss 
        << FormLocation()
        << relpath 
        << "+" 
        << line
        ; 
    return ss.str();
}


// NB not all invokations are instrumented, 
//    ie there are some internal calls to flat
//    and some external, so distinguish with m_internal
//
double CRandomEngine::flat_instrumented(const char* file, int line)
{
    m_location = FormLocation(file, line);
    m_internal = true ; 
    double _flat = flat();
    m_internal = false ;
    return _flat ; 
}

double CRandomEngine::flat() 
{ 
    if(!m_internal) m_location = FormLocation();

    m_locseq->add(m_location.c_str(), m_ctx._record_id, m_ctx._step_id); 

    double _flat =  m_engine->flat() ;  


    G4VProcess* proc = CProcess::CurrentProcess() ; 

    CSteppingState ss = CStepping::CurrentState(); 

    std::cerr 
        << desc()
        << " " 
        << std::setw(20) << CStepStatus::Desc(ss.fStepStatus)
        << " " << CProcess::Desc(proc)       
        <<  std::endl 
        ; 


/*
    std::cerr 
          << " flat " << std::setw(10) << _flat 
          << desc()
          << std::endl 
          ; 
*/

    return _flat ; 
}


void CRandomEngine::poststep()
{
    m_locseq->poststep();
}

void CRandomEngine::posttrack()
{
    unsigned long long seqhis = m_g4->getSeqHis()  ;
    m_locseq->mark(seqhis);
}

void CRandomEngine::dump(const char* msg) const 
{
    m_locseq->dump(msg);
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

