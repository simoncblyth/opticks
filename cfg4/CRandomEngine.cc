
#include "PLOG.hh"

#include "SDigest.hh"
#include "SSys.hh"
#include "BStr.hh"
#include "BFile.hh"
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
    m_internal(false),
    m_james(new CLHEP::HepJamesRandom()),
    m_nonran(new CLHEP::NonRandomEngine()),
    m_engine(m_james),
    m_count(0),
    m_count_mismatch(0),
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


std::string CRandomEngine::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " record_id " << std::setw(5) << m_ctx._record_id 
       << " count " << std::setw(5) << m_count
       << " step_id " << std::setw(5) << m_ctx._step_id
       << " loc " << m_location 
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

    m_location_vec.push_back(m_location); 

    double _flat =  m_engine->flat() ;  

   /*
    std::cerr 
          << " flat " << std::setw(10) << _flat 
          << desc()
          << std::endl 
          ; 
   */ 


    m_record_count[m_ctx._record_id]++ ; 

    if( int(m_count) == m_harikari ) 
    {
        LOG(error) << "OPTICKS_CRANDOMENGINE_HARIKARI" ; 
        assert(0) ;  
    }

    m_count++ ; 

    return _flat ; 
}


void CRandomEngine::posttrack()
{
    m_digest = SDigest::digest(m_location_vec); 
    m_digest_locations[m_digest] = BStr::join(m_location_vec, ',') ;

    unsigned long long seqmat = m_g4->getSeqMat()  ;
    unsigned long long seqhis = m_g4->getSeqHis()  ;

/*
    LOG(info) 
              << " record_id " << m_ctx._record_id
              << " m_location_vec.size() " << m_location_vec.size()
              << " digest " << m_digest  
              << " seqhis " << std::hex << seqhis << std::dec
              << " seqmat " << std::hex << seqmat << std::dec
              ;
*/
    // TODO : flip map k<->v the location digest is 
    //        more unique that the seqhis
    //        getting non-uniqueness at 31/100k level

    if(m_seqhis_digest.count(seqhis) == 0)
    {
        m_seqhis_digest[seqhis] = m_digest ; 
    }
    else if(m_seqhis_digest.count(seqhis) == 1)
    {
        std::string prior = m_seqhis_digest[seqhis] ;  
        bool match = m_digest.compare(prior.c_str()) == 0 ;
        if(!match) 
        {
           LOG(error) 
              << " record_id " << m_ctx._record_id
              << " m_location_vec.size() " << m_location_vec.size()
              << " digest " << m_digest  
              << " seqhis " << std::hex << seqhis << std::dec
              << " seqmat " << std::hex << seqmat << std::dec
              << " digest/seqhis non-uniqueness " 
              << " prior " << prior
              << " count_mismatch " << m_count_mismatch
              ;

            m_count_mismatch++ ; 
            for(unsigned i=0 ; i < m_location_vec.size() ; i++ ) std::cerr << m_location_vec[i] << std::endl ;  

        }
        //assert( match );
    }
    else
    {
        assert(0);  
    }

    m_location_vec.clear();
}


void CRandomEngine::dump(const char* msg) const 
{
    LOG(info) << msg ; 



    typedef std::map<unsigned, unsigned> UMM ; 
    for(UMM::const_iterator it=m_record_count.begin() ; it != m_record_count.end() ; it++ )
    {
        unsigned count = it->second ; 
        if(count > 17)
        std::cout 
            << std::setw(10) << it->first 
            << std::setw(10) << count
            << std::endl 
            ;
    }


    typedef std::map<unsigned long long, std::string> ULLSM ; 
    for(ULLSM::const_iterator it=m_seqhis_digest.begin() ; it != m_seqhis_digest.end() ; it++ )
    {
        std::cout 
            << " seqhis " << std::setw(16) << std::hex << it->first << std::dec
            << " digest " << std::setw(32) << it->second
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



