
#include <array>

#include "PLOG.hh"

#include "Randomize.hh"
#include "CLHEP/Random/NonRandomEngine.h"
#include "G4String.hh"
#include "G4VProcess.hh"


#include "SPairVec.hh"
#include "SMap.hh"
#include "SDigest.hh"
#include "SSys.hh"

#include "BStr.hh"
#include "BFile.hh"

#include "Opticks.hh"

#include "CG4.hh"
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
    m_skipdupe(true),
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

    bool is_skip = m_skipdupe && m_location_vec.size() > 0 && m_location_vec.back().compare(m_location.c_str()) == 0 ; 
    if(!is_skip)
    {
        m_location_vec.push_back(m_location); 
    } 


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
    //m_digest = m_skipdupe ? SDigest::digest_skipdupe(m_location_vec) : SDigest::digest(m_location_vec); 
    m_digest = SDigest::digest(m_location_vec); 
    m_digest_count[m_digest]++ ; 
    m_digest_locations[m_digest] = BStr::join(m_location_vec, ',') ;
    m_location_vec.clear();

    //unsigned long long seqmat = m_g4->getSeqMat()  ;
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

    //  initially keyed on seqhis, but found   
    //  that some seqhis yielding multiple digests
    //  at 31/100k level
    //
    //  instead keying on the digest, have so far found that 
    //  every digest (sequence of code locations) always has the same unique seqhis

    unsigned digest_seqhis_count = m_digest_seqhis.count(m_digest) ;

    if(digest_seqhis_count == 0) // fresh digest (sequence of code locations)
    {
        m_digest_seqhis[m_digest] = seqhis ; 
    }
    else if(digest_seqhis_count == 1)  // repeated digest 
    {
        unsigned long long prior_seqhis = m_digest_seqhis[m_digest] ; 
        bool match = prior_seqhis == seqhis ; 
        assert(match) ; 
    }
    else
    {
        assert(0 && "NEVER : would indicate std::map key failure");  
    }
}



void CRandomEngine::dumpDigests(const char* msg, bool locations) const 
{ 
    LOG(info) << msg ; 
    typedef std::string K ; 
    typedef unsigned long long V ; 
    typedef std::map<K, V> MKV ; 

    typedef std::pair<K, unsigned> PKU ;  
    typedef std::vector<PKU> LPKU ; 

    LPKU digest_counts ;

    unsigned total(0) ; 
    for(MKV::const_iterator it=m_digest_seqhis.begin() ; it != m_digest_seqhis.end() ; it++ ) 
    { 
        K digest = it->first ;
        unsigned count = m_digest_count.at(digest) ; 
        digest_counts.push_back(PKU(digest, count)) ; 
        total += count ; 
    }

    bool ascending = false ; 
    SPairVec<K, unsigned> spv(digest_counts, ascending);
    spv.sort();


    std::cout 
        << " total "    << std::setw(10) << total
        << " skipdupe " << ( m_skipdupe ? "Y" : "N" ) 
        << std::endl 
        ;


    for(LPKU::const_iterator it=digest_counts.begin() ; it != digest_counts.end() ; it++ )
    {
        PKU digest_count = *it ;
        K digest = digest_count.first ; 
        unsigned count = digest_count.second ;  

        V seqhis = m_digest_seqhis.at(digest) ;

        unsigned num_digest_with_seqhis = SMap<K,V>::ValueCount(m_digest_seqhis, seqhis );         

        std::vector<K> k_digests ; 
        SMap<K,V>::FindKeys(m_digest_seqhis, k_digests, seqhis, false );
        assert( k_digests.size() == num_digest_with_seqhis );
  
        std::cout 
            << " count "    << std::setw(10) << count
            << " k:digest " << std::setw(32) << digest
            << " v:seqhis " << std::setw(32) << std::hex << seqhis << std::dec
            << " num_digest_with_seqhis " << std::setw(10) << num_digest_with_seqhis
            << std::endl 
            ;

        assert( num_digest_with_seqhis > 0 );
        if( num_digest_with_seqhis > 1 && locations ) dumpLocations( k_digests );  
    }
}



void CRandomEngine::dumpLocations( const std::vector<std::string>& digests ) const 
{
    typedef std::vector<std::string> VS ; 

    VS* tab = new VS[digests.size()] ;   

    unsigned ndig = digests.size() ;
    unsigned nmax = 0 ; 

    for(unsigned i=0 ; i < ndig ; i++)
    {
         std::string dig = digests[i] ; 
         std::string locs = m_digest_locations.at(dig) ; 

         VS& locv = tab[i] ; 
         BStr::split(locv, locs.c_str(), ',' ); 
         if( locv.size() > nmax ) nmax = locv.size() ; 
    }

    LOG(info) << "dumpLocations"
              << " ndig " << ndig 
              << " nmax " << nmax
              ; 

    for( unsigned j=0 ; j < nmax ; j++ ) 
    {
        for( unsigned i=0 ; i < ndig ; i++ ) 
        { 
            VS& loci = tab[i] ; 
            std::cerr << std::setw(50) << ( j < loci.size() ? loci[j] : "-" ) ;
        }
        std::cerr  << std::endl ;
    }

    delete [] tab ; 

/*
 k:digest 274ceb8e0097317bfd3e25c4cc70b714 v:seqhis                            86ccd num_digest_with_seqhis          3
2017-12-06 12:58:24.530 INFO  [482288] [CRandomEngine::dumpLocations@249] dumpLocations ndig 3 nmax 30
                                    Scintillation;                                    Scintillation;                                    Scintillation;
                                       OpBoundary;                                       OpBoundary;                                       OpBoundary;
                                       OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                     OpAbsorption;                                     OpAbsorption;                                     OpAbsorption;
     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                    Scintillation;                                    Scintillation;                                    Scintillation;
                                       OpBoundary;                                       OpBoundary;                                       OpBoundary;
     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                    Scintillation;                                    Scintillation;                                    Scintillation;
                                       OpBoundary;                                       OpBoundary;                                       OpBoundary;
                                       OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                       OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                       OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                       OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                       OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                    Scintillation;                                       OpRayleigh;                                       OpRayleigh;
                                       OpBoundary;                                       OpRayleigh;                                       OpRayleigh;
                                       OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655                                       OpRayleigh;                                       OpRayleigh;
       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269                                       OpRayleigh;                                       OpRayleigh;
                                                 -                                       OpRayleigh;                                    Scintillation;
                                                 -                                       OpRayleigh;                                       OpBoundary;
                                                 -                                       OpRayleigh;                                       OpRayleigh;
                                                 -                                       OpRayleigh;      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
                                                 -                                       OpRayleigh;       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
                                                 -                                    Scintillation;                                                 -
                                                 -                                       OpBoundary;                                                 -
                                                 -                                       OpRayleigh;                                                 -
                                                 -      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655                                                 -
                                                 -       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269                                                 -
*/



    // TODO: ekv recording to retain step splits 
}



void CRandomEngine::dumpCounts(const char* msg) const 
{ 
    LOG(info) << msg ; 
    typedef std::map<unsigned, unsigned> UMM ; 
    for(UMM::const_iterator it=m_record_count.begin() ; it != m_record_count.end() ; it++ )
    {
        unsigned count = it->second ; 
        if(count > 50)
        std::cout 
            << std::setw(10) << it->first 
            << std::setw(10) << count
            << std::endl 
            ;
    }
}

 

void CRandomEngine::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    dumpCounts(msg); 
    dumpDigests(msg, false); 
    dumpDigests(msg, true); 
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



