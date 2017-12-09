
#include <iomanip>

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

#include "NPY.hpp"

#include "CG4.hh"
#include "CProcess.hh"
#include "CProcessManager.hh"
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
    m_mask(m_ok->getMask()),
    m_masked(m_mask.size() > 0),
    m_path("$TMP/TRngBufTest.npy"),
    m_alignlevel(m_ok->getAlignLevel()),
    m_seed(9876),
    m_internal(false),
    m_skipdupe(true),
    m_locseq(m_alignlevel > 1 ? new BLocSeq<unsigned long long>(m_skipdupe) : NULL ),
    m_james(new CLHEP::HepJamesRandom()),
    m_nonran(new CLHEP::NonRandomEngine()),
    m_engine(m_nonran),
    m_curand(NPY<double>::load(m_path)),
    m_curand_ni(m_curand ? m_curand->getShape(0) : 0 ),
    m_curand_nv(m_curand ? m_curand->getNumValues(1) : 0 ),
    m_current_record_flat_count(0),
    m_flat(-1.0)
{
    init();
}


bool CRandomEngine::hasSequence() const 
{
    return m_curand && m_curand_ni > 0 && m_curand_nv > 0 ; 
}

const char* CRandomEngine::getPath() const 
{
    return m_path ; 
}

void CRandomEngine::dumpDouble(const char* msg, double* v, unsigned width ) const 
{
    LOG(info) << msg ; 
    assert( m_curand_nv > 15 );
    for(int i=0 ; i < 16 ; i++)  
    {
        std::cout << std::fixed << std::setw(10) << std::setprecision(10) << v[i] << " " ; 
        if( i % width == (width - 1) ) std::cout << std::endl ; 
    }
}


void CRandomEngine::init()
{
    initCurand();
    CLHEP::HepRandom::setTheEngine( this );  
    CLHEP::HepRandom::setTheSeed(  m_seed );    // does nothing for NonRandom
}



void CRandomEngine::initCurand()
{
    LOG(info) << ( m_curand ? m_curand->getShapeString() : "-" ) 
              << " curand_ni " << m_curand_ni
              << " curand_nv " << m_curand_nv
              ; 

    if(!m_curand) return ; 
        
    unsigned w = 4 ; 
    if( m_curand_ni > 0 )
         dumpDouble( "v0" , m_curand->getValues(0), w ) ; 

    if( m_curand_ni > 1 )
         dumpDouble( "v1" , m_curand->getValues(1), w ) ; 

    if( m_curand_ni > 99999 )
        dumpDouble( "v99999" , m_curand->getValues(99999), w ) ; 
    
}

void CRandomEngine::setupCurandSequence(int record_id)
{
    assert( isNonRan() ) ; 

    assert( record_id > -1 && record_id < m_curand_ni ); 

    assert( m_curand_nv > 0 ) ;

    double* seq = m_curand->getValues(record_id) ; 

    m_nonran->setRandomSequence( seq, m_curand_nv ) ; 

    m_current_record_flat_count = 0 ; 
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

    std::stringstream rs1_ ; 
    rs1_ <<  m_ctx._record_id  << "." << ( m_ctx._step_id + 1 ) ; 
    std::string rs1 = rs1_.str() ; 

    std::stringstream ss ; 
    ss 
       << "CRandomEngine"
       << " rec.stp1 " << std::setw(5) << rs1
       << " crfc " << std::setw(5) << m_current_record_flat_count 
       << " loc " << std::setw(50) << m_location 
       ;

    return ss.str();
}


std::string CRandomEngine::FormLocation()
{
    G4VProcess* proc = CProcess::CurrentProcess() ; 

    std::stringstream ss ; 
    ss <<  ( proc ? proc->GetProcessName().c_str() : "NoProc" )
       << ";" 
       ;  

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

    assert( m_current_record_flat_count < m_curand_nv ); 

    m_flat =  m_engine->flat() ;  

    if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 

    m_current_record_flat_count++ ; 
 
    return m_flat ; 
}


void CRandomEngine::dumpFlat()
{
    // locseq was just for development, not needed in ordinary usage
    if(m_locseq)
    m_locseq->add(m_location.c_str(), m_ctx._record_id, m_ctx._step_id); 

    G4VProcess* proc = CProcess::CurrentProcess() ; 
    CSteppingState ss = CStepping::CurrentState(); 
    std::cerr 
        << desc()
        << " "
        << std::setw(10) << m_flat 
        << " " 
        << std::setw(20) << CStepStatus::Desc(ss.fStepStatus)
        << " " << CProcess::Desc(proc)       
        << " alignlevel " << m_alignlevel
        <<  std::endl 
        ; 
}


void CRandomEngine::poststep()
{
    if( m_locseq )
    {
        m_locseq->poststep();
        LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ; 
    }
}


// invoked from CG4::pretrack following CG4Ctx::setTrack
void CRandomEngine::pretrack()
{
    // where is a better place to do this ? maybe CG4Ctx::setTrack
    unsigned index = m_ctx._record_id   ;
    if(m_mask.size() > 0)
    {
        assert( index < m_mask.size() );
        index = m_mask[index] ; 
    }

    LOG(error) << "CRandomEngine::pretrack record_id: " 
               << " ctx.record_id " << m_ctx._record_id 
               << " index " << index 
               << " mask.size " << m_mask.size()
               ;
    
    setupCurandSequence(index) ;
}


void CRandomEngine::posttrack()
{
    if(m_locseq)
    {
        unsigned long long seqhis = m_g4->getSeqHis()  ;
        m_locseq->mark(seqhis);
        LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ; 
    }
}

void CRandomEngine::dump(const char* msg) const 
{
    if(!m_locseq) return ; 
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

