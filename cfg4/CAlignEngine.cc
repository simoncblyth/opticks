#include <sstream>
#include <cassert>
#include "Randomize.hh"

#include "BFile.hh"
#include "PLOG.hh"
#include "NPY.hpp"
#include "SBacktrace.hh"
#include "CAlignEngine.hh"

const char* CAlignEngine::LOGNAME = "CAlignEngine.log" ; 
CAlignEngine* CAlignEngine::INSTANCE = NULL ; 

bool CAlignEngine::Initialize(const char* simstreamdir ) // static
{
    if(INSTANCE == NULL ) 
    {
        //const char* reldir = PLOG::instance->exename() ; 
        INSTANCE = new CAlignEngine(simstreamdir) ; 
    }
    return INSTANCE->isReady(); 
}

void CAlignEngine::Finalize() // static
{
    delete INSTANCE ;  
}


void CAlignEngine::SetSequenceIndex(int seq_index) // static
{
    if(INSTANCE == NULL ) INSTANCE = new CAlignEngine(NULL) ; 
    INSTANCE->setSequenceIndex(seq_index); 
}

const char* CAlignEngine::InitSimLog( const char* ssdir) // static
{
    if( ssdir == NULL ) return NULL ; 
    std::string path = BFile::preparePath( ssdir, LOGNAME ); 
    return strdup(path.c_str()); 
}

CAlignEngine::~CAlignEngine()
{
    if(!m_sslogpath) return ; 
    
    std::string path = BFile::ChangeExt( m_sslogpath, ".npy" ); 
    LOG(info) << " saving cursors to " << path ; 
    m_cur->save(path.c_str()); 
}


CAlignEngine::CAlignEngine(const char* ssdir)
    :
    m_seq_path("$TMP/TRngBufTest.npy"),
    m_seq(NPY<double>::load(m_seq_path)),
    m_seq_values(m_seq ? m_seq->getValues() : NULL),
    m_seq_ni(m_seq ? m_seq->getShape(0) : 0 ),
    m_seq_nv(m_seq ? m_seq->getNumValues(1) : 0 ),  // itemvalues
    m_cur(NPY<int>::make(m_seq_ni)),
    m_cur_values(m_cur->fill(0)),
    m_seq_index(-1),
    m_recycle(true),
    m_default(CLHEP::HepRandom::getTheEngine()),
    m_sslogpath(InitSimLog(ssdir)),
    m_backtrace(true),
    m_out(NULL),
    m_count(0),
    m_modulo(1000)
{
    assert( m_default ); 
    LOG(info) << desc(); 

    bool has_seq = m_seq_ni > 0 ; 
    if(!has_seq) LOG(fatal) << "MISSING/EMPTY m_seq_path : " << m_seq_path 
                            << " ( Run TRngBufTest executable to generate the missing .npy file ) " << desc() ; 

    assert(has_seq); 

    if(!m_backtrace) return ; 

    if(m_sslogpath) 
    { 
        m_out = new std::ofstream(m_sslogpath) ;
        LOG(info) << " simstream logpath " << m_sslogpath ; 
    }
    else
    {
        m_out = new std::ostream(std::cout.rdbuf());
    }
    (*m_out) << desc() << std::endl ;  
}

bool CAlignEngine::isReady() const 
{
    return m_seq && m_seq_ni > 0 && m_seq_nv > 0 ; 
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
       << " simstream logpath " << ( m_sslogpath ? m_sslogpath : "-" )
       << " recycle_idx " << m_recycle_idx.size()
       ;
    return ss.str(); 
}


void CAlignEngine::setSequenceIndex(int seq_index)
{
    bool have_seq = seq_index < m_seq_ni ; 
    if(!have_seq) LOG(fatal) << "OUT OF RANGE : " << desc() ; 
    assert( have_seq );
  
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


   
    if( m_recycle == false )
    {
        assert( cursor < m_seq_nv ) ; 
    }
    else
    {
        // note that this does not change the value of the cursor 
        // in the m_cur buffer : it just cycles the usage if it 

        if(cursor >= m_seq_nv ) 
        {
            unsigned n0 = m_recycle_idx.size(); 
            m_recycle_idx.insert(m_seq_index); 
            unsigned n1 = m_recycle_idx.size(); 

            if( n1 > n0) 
            LOG(error) << " recycling RNG : not enough pre-cooked :" 
                       << " seq_index " << m_seq_index 
                       << " seq_nv " << m_seq_nv
                       << " cursor " << cursor
                       << " recycle_idx " << m_recycle_idx.size()
                       ; 
 
            cursor = cursor % m_seq_nv ;  
        }
    }


    int idx = m_seq_index*m_seq_nv + cursor ; 

    double u = m_seq_values[idx] ; 


    if( m_backtrace )
    {
        //bool first = cursor == 0 && m_seq_index == 0 ;
        //if(first) SBacktrace::Dump(*m_out);  

        const char* caller = SBacktrace::CallSite( "::flat" ) ;


        if( m_modulo == 0 || m_count % m_modulo == 0 ) 
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

    m_count++ ; 

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

