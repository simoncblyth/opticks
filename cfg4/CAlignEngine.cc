#include <sstream>
#include <cassert>
#include "Randomize.hh"

#include "BFile.hh"
#include "PLOG.hh"
#include "NPY.hpp"
#include "SBacktrace.hh"
#include "CAlignEngine.hh"

CAlignEngine* CAlignEngine::INSTANCE = NULL ; 

void CAlignEngine::Initialize(const char* simstreampath ) // static
{
    if(INSTANCE == NULL ) INSTANCE = new CAlignEngine(simstreampath) ; 
}
void CAlignEngine::SetSequenceIndex(int seq_index) // static
{
    if(INSTANCE == NULL ) INSTANCE = new CAlignEngine(NULL) ; 
    INSTANCE->setSequenceIndex(seq_index); 
}

CAlignEngine::CAlignEngine(const char* simstreampath)
    :
    m_seq_path("$TMP/TRngBufTest.npy"),
    m_seq(NPY<double>::load(m_seq_path)),
    m_seq_values(m_seq ? m_seq->getValues() : NULL),
    m_seq_ni(m_seq ? m_seq->getShape(0) : 0 ),
    m_seq_nv(m_seq ? m_seq->getNumValues(1) : 0 ),  // itemvalues
    m_cur(NPY<int>::make(m_seq_ni)),
    m_cur_values(m_cur->fill(0)),
    m_seq_index(-1),
    m_default(CLHEP::HepRandom::getTheEngine()),
    m_simstreampath(simstreampath ? strdup(simstreampath) : NULL),
    m_backtrace(true),
    m_out(NULL)
{
    assert( m_default ); 
    LOG(info) << desc(); 

    if(!m_backtrace) return ; 

    if(m_simstreampath) 
    { 
        std::string path = BFile::preparePath( m_simstreampath ); 
        m_out = new std::ofstream(path.c_str()) ;
        LOG(info) << " simstreampath " << path ; 
    }
    else
    {
        m_out = new std::ostream(std::cout.rdbuf());
    }
    (*m_out) << desc() << std::endl ;  
}

std::string CAlignEngine::desc() const 
{
    std::stringstream ss ; 
    ss << name()
       << " seq_index " << m_seq_index 
       << " seq " << ( m_seq ? m_seq->getShapeString() : "-" )
       << " seq_ni " << m_seq_ni
       << " seq_nv " << m_seq_nv
       << " cur " << ( m_cur ? m_cur->getShapeString() : "-" )
       << " seq_path " << ( m_seq_path ? m_seq_path : "-" )
       << " simstreampath " << ( m_simstreampath ? m_simstreampath : "-" )
       ;
    return ss.str(); 
}

void CAlignEngine::setSequenceIndex(int seq_index)
{
    assert( seq_index < m_seq_ni );
  
    m_seq_index = seq_index ; 

    if( m_seq_index < 0) 
    {
        disable(); 
    }
    else 
    {
        enable();
    }
}

bool CAlignEngine::isTheEngine() const 
{
    return this == CLHEP::HepRandom::getTheEngine() ; 
}

void CAlignEngine::enable() const 
{
    if(!isTheEngine())
    {
        const CAlignEngine* this0 = this ; 
        CAlignEngine* this1 = const_cast<CAlignEngine*>(this0);  
        CLHEP::HepRandom::setTheEngine( dynamic_cast<CLHEP::HepRandomEngine*>(this1) );  
    }
}

void CAlignEngine::disable() const 
{
    if(isTheEngine())
    {
         CLHEP::HepRandom::setTheEngine( m_default );  
    } 
    else
    {
         LOG(debug) << " cannot disable as are not currently theEngine " ;  
    }
}


double CAlignEngine::flat() 
{
    if(m_seq_index < 0)
    {
        assert( 0 && " should not be called whilst disabled, use G4UniformRand() to get from the encumbent engine  " ) ; 
        return 0 ;
    }

    int cursor = *(m_cur_values + m_seq_index) ;

    *(m_cur_values + m_seq_index) += 1 ;   

    assert( cursor < m_seq_nv ) ; 

    int idx = m_seq_index*m_seq_nv + cursor ; 

    double u = m_seq_values[idx] ; 


    if( m_backtrace )
    {
        bool first = cursor == 0 && m_seq_index == 0 ;

        if(first) SBacktrace::Dump(*m_out);  

        const char* caller = SBacktrace::CallSite( "::flat" ) ;

        (*m_out) 
            << "(" 
            << std::setw(6) << m_seq_index 
            << ":"
            << std::setw(4) << cursor  
            << ") "
            << std::setw(10) << std::fixed << u
            << " : "
            << caller
            << std::endl 
            ;   
    }

    return u ; 
}

std::string CAlignEngine::name() const 
{
    return "CAlignEngine" ; 
}

void CAlignEngine::flatArray(const int, double* ) 
{
    assert(0);
}
void CAlignEngine::setSeed(long, int ) 
{
    assert(0);
} 
void CAlignEngine::setSeeds(const long *, int) 
{
    assert(0);
}
void CAlignEngine::saveStatus( const char * ) const 
{
    assert(0);
}       
void CAlignEngine::restoreStatus( const char * )
{
    assert(0);
}
void CAlignEngine::showStatus() const 
{
    assert(0);
}

