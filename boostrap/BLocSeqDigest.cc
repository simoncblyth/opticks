#include <cassert>
#include <iostream>
#include "PLOG.hh"

#include "SPairVec.hh"
#include "SMap.hh"
#include "SDigest.hh"


#include "BStr.hh"
#include "BLocSeqDigest.hh"


template <typename T>
BLocSeqDigest<T>::BLocSeqDigest(bool skipdupe, bool requirematch, unsigned dump_loc_min) 
    :
    _skipdupe(skipdupe),
    _requirematch(requirematch),
    _dump_loc_min(dump_loc_min)
{
}

template <typename T>
void BLocSeqDigest<T>::add(const char* loc)
{
    bool is_skip = _skipdupe && _locs.size() > 0 && _locs.back().compare(loc) == 0 ; 
    if(!is_skip)
    {
        _locs.push_back(loc); 
    } 
}

template <typename T>
void BLocSeqDigest<T>::mark(T marker)
{
    std::string digest = SDigest::digest(_locs); 

    _digest_count[digest]++ ; 
    _digest_locations[digest] = BStr::join(_locs, ',') ;
    _locs.clear();

    unsigned digest_marker_count = _digest_marker.count(digest) ;

    if(digest_marker_count == 0) // fresh digest (sequence of code locations)
    {
        _digest_marker[digest] = marker ; 
    }
    else if(digest_marker_count == 1)  // repeated digest 
    {
        T prior_marker = _digest_marker[digest] ; 
        bool match = prior_marker == marker ; 

        if(!_requirematch) return ; 
        if(!match)
        {
            LOG(info) 
                << " prior_marker " << std::hex << prior_marker << std::dec
                << " marker " << std::hex << marker << std::dec
                ;
        }
        assert(match) ; 
    }
    else
    {
        assert(0 && "NEVER : would indicate std::map key failure");  
    }
}





template <typename T>
void BLocSeqDigest<T>::dumpDigests(const char* msg, bool locations) const 
{ 
    LOG(info) << msg ; 
    typedef std::string K ; 
    typedef std::map<K, T> MKV ; 

    typedef std::pair<K, unsigned> PKU ;  
    typedef std::vector<PKU> LPKU ; 

    LPKU ordering_digest_counts ;

    unsigned total(0) ; 
    for(typename MKV::const_iterator it=_digest_marker.begin() ; it != _digest_marker.end() ; it++ ) 
    { 
        K digest = it->first ;
        unsigned count = _digest_count.at(digest) ; 
        ordering_digest_counts.push_back(PKU(digest, count)) ; 
        total += count ; 
    }

    bool ascending = false ; 
    SPairVec<K, unsigned> spv(ordering_digest_counts, ascending);
    spv.sort();

    std::cout 
        << " total "    << std::setw(10) << total
        << " skipdupe " << ( _skipdupe ? "Y" : "N" ) 
        << std::endl 
        ;

    for(LPKU::const_iterator it=ordering_digest_counts.begin() ; it != ordering_digest_counts.end() ; it++ )
    {
        PKU digest_count = *it ;
        K digest = digest_count.first ; 
        unsigned count = digest_count.second ;  

        T marker = _digest_marker.at(digest) ;

        unsigned num_digest_with_marker = SMap<K,T>::ValueCount(_digest_marker, marker );         

        std::vector<K> k_digests ; 
        SMap<K,T>::FindKeys(_digest_marker, k_digests, marker, false );
        assert( k_digests.size() == num_digest_with_marker );
  
        std::cout 
            << " count "    << std::setw(10) << count
            << " k:digest " << std::setw(32) << digest
            << " v:marker " << std::setw(32) << std::hex << marker << std::dec
            << " num_digest_with_marker " << std::setw(10) << num_digest_with_marker
            << std::endl 
            ;

        assert( num_digest_with_marker > 0 );
        if( num_digest_with_marker > _dump_loc_min && locations ) dumpLocations( k_digests );  
    }
}


template <typename T>
void BLocSeqDigest<T>::dumpLocations( const std::vector<std::string>& digests ) const 
{
    typedef std::vector<std::string> VS ; 

    VS* tab = new VS[digests.size()] ;   

    unsigned ndig = digests.size() ;
    unsigned nmax = 0 ; 

    for(unsigned i=0 ; i < ndig ; i++)
    {
         std::string dig = digests[i] ; 
         std::string locs = _digest_locations.at(dig) ; 

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
}


template struct BLocSeqDigest<unsigned long long>;


