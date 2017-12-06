#include <cassert>
#include <iostream>
#include "PLOG.hh"

#include "SPairVec.hh"
#include "SMap.hh"
#include "SDigest.hh"


#include "BStr.hh"
#include "BLocSeq.hh"


template <typename T>
BLocSeq<T>::BLocSeq(bool skipdupe)
    :
    m_skipdupe(skipdupe),
    m_global_flat_count(0),
    m_step_flat_count(0),
    m_count_mismatch(0)
{
}

template <typename T>
void BLocSeq<T>::add(const char* loc, int record_id, int step_id )
{
    assert( step_id >= -1 ) ; 
    bool is_skip = m_skipdupe && m_location_vec.size() > 0 && m_location_vec.back().compare(loc) == 0 ; 
    if(!is_skip)
    {
        m_location_vec.push_back(loc); 
    } 

    m_record_count[record_id]++ ; 
    m_global_flat_count++ ; 
    m_step_flat_count++ ; 
}


template <typename T>
void BLocSeq<T>::poststep()
{
/*
    LOG(info) 
         << " global_flat_count " << m_global_flat_count 
         << " step_flat_count " << m_step_flat_count 
         ; 
*/
    m_step_flat_count = 0 ; 
}




template <typename T>
void BLocSeq<T>::mark(T marker)
{
    std::string digest = SDigest::digest(m_location_vec); 
    m_digest_count[digest]++ ; 
    m_digest_locations[digest] = BStr::join(m_location_vec, ',') ;
    m_location_vec.clear();


    unsigned digest_marker_count = m_digest_marker.count(digest) ;

    if(digest_marker_count == 0) // fresh digest (sequence of code locations)
    {
        m_digest_marker[digest] = marker ; 
    }
    else if(digest_marker_count == 1)  // repeated digest 
    {
        T prior_marker = m_digest_marker[digest] ; 
        bool match = prior_marker == marker ; 
        assert(match) ; 
    }
    else
    {
        assert(0 && "NEVER : would indicate std::map key failure");  
    }
}






template <typename T>
void BLocSeq<T>::dump(const char* msg) const 
{
    dumpCounts(msg); 
    dumpDigests(msg, false); 
    dumpDigests(msg, true); 
}

template <typename T>
void BLocSeq<T>::dumpCounts(const char* msg) const 
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

template <typename T>
void BLocSeq<T>::dumpDigests(const char* msg, bool locations) const 
{ 
    LOG(info) << msg ; 
    typedef std::string K ; 
    typedef std::map<K, T> MKV ; 

    typedef std::pair<K, unsigned> PKU ;  
    typedef std::vector<PKU> LPKU ; 

    LPKU digest_counts ;

    unsigned total(0) ; 
    for(typename MKV::const_iterator it=m_digest_marker.begin() ; it != m_digest_marker.end() ; it++ ) 
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

        T marker = m_digest_marker.at(digest) ;

        unsigned num_digest_with_marker = SMap<K,T>::ValueCount(m_digest_marker, marker );         

        std::vector<K> k_digests ; 
        SMap<K,T>::FindKeys(m_digest_marker, k_digests, marker, false );
        assert( k_digests.size() == num_digest_with_marker );
  
        std::cout 
            << " count "    << std::setw(10) << count
            << " k:digest " << std::setw(32) << digest
            << " v:marker " << std::setw(32) << std::hex << marker << std::dec
            << " num_digest_with_marker " << std::setw(10) << num_digest_with_marker
            << std::endl 
            ;

        assert( num_digest_with_marker > 0 );
        if( num_digest_with_marker > 1 && locations ) dumpLocations( k_digests );  
    }
}



template <typename T>
void BLocSeq<T>::dumpLocations( const std::vector<std::string>& digests ) const 
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









template class BLocSeq<unsigned long long>;

