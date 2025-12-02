#pragma once
/**
sutil.h
========

For testing hit merging it is useful to
start from the intended merged result
and unmerge that to provide an input to
merge tests that has known expected result.

Hits are merged within (identity, timebucket)
ie the merged hits all have the same identity
and timebucket with merge result time being the
earliest time.


sutil::sort_photon_by_key
    as the process of merging works by sorting, need to sort
    to properly mockup the result of merging

sutil::mock_merged
    artificially set up a merged result with hitcount
    and identity values which is then sorted by key

sutil::unmerge
    unmerging the mock_merged photons produces a test
    sample of photons with known expected merge result

**/


#include <algorithm>
#include <numeric>
#include <random>

#include "OpticksPhoton.h"
#include "NP.hh"
#include "sphoton.h"
#include "sphotonlite.h"


struct sutil
{
    template<typename T>
    static void check_shape(const NP* a);

    template<typename T>
    static bool is_lite();

    template<typename T>
    static std::string photon_name(const char* prefix=nullptr, const char* suffix=nullptr);

    template<typename T>
    static std::string hit_name(const char* prefix=nullptr, const char* suffix=nullptr, bool merged=false);

    template<typename T>
    static NP* sort_photon_by_key(const NP* photon, float time_window );

    template<typename T>
    static NP* mock_merged(size_t ni, float time_window);

    template<typename T>
    static NP* unmerge(const NP* photon);

    template<typename T>
    static NP* shuffle(const NP* photon, uint64_t seed=42 );


    static void create_photonlite_from_photon( sphotonlite& lite, const sphoton& p );
    static NP*  create_photonlite_from_photon( const NP* photon );

};

template<typename T>
inline void sutil::check_shape(const NP* a)
{
    size_t ni = a->num_items();

    if(strcmp(T::NAME, "sphoton") == 0)
    {
        assert(a->has_shape(ni, 4, 4));
    }
    else if(strcmp(T::NAME, "sphotonlite") == 0)
    {
        assert(a->has_shape(ni, 4));
    }
    else
    {
        assert(0); // unexpected T::NAME
    }
}


template<typename T>
inline bool sutil::is_lite()
{
    bool lite = false ;
    if(     strcmp(T::NAME, "sphoton") == 0)     lite = false ;
    else if(strcmp(T::NAME, "sphotonlite") == 0) lite = true ;
    return lite ;
}

template<typename T>
inline std::string sutil::photon_name(const char* prefix, const char* suffix)
{
    std::stringstream ss;
    if(prefix) ss << prefix ;
    ss << "photon" ;
    ss << ( is_lite<T>() ? "lite" : "" ) ;
    if(suffix) ss << suffix ;
    std::string name = ss.str();
    return name ;
}

template<typename T>
inline std::string sutil::hit_name(const char* prefix, const char* suffix, bool merged)
{
    std::stringstream ss;
    if(prefix) ss << prefix ;
    ss << "hit" ;
    ss << ( is_lite<T>() ? "lite" : "" ) ;
    if(merged) ss << "merged" ;
    if(suffix) ss << suffix ;
    std::string name = ss.str();
    return name ;
}


template<typename T>
inline NP* sutil::sort_photon_by_key(const NP* photon, float time_window)
{
    check_shape<T>(photon);
    size_t ni = photon->num_items();

    const T* src = (const T*)photon->bytes();
    NP* sorted = T::zeros(ni);
    T* dst = (T*)sorted->bytes();

    // Create index array
    std::vector<unsigned> idx(ni);
    std::iota(idx.begin(), idx.end(), 0u);  // fill 0,1,2,...,ni-1

    using key_functor = typename T::key_functor ;
    key_functor key_fn {time_window};

    // Sort indices by the computed key
    std::stable_sort(idx.begin(), idx.end(),
        [&](unsigned a, unsigned b) -> bool {
            return key_fn(src[a]) < key_fn(src[b]);
        });

    // Gather sorted photons
    for (size_t i = 0; i < ni; ++i) dst[i] = src[idx[i]];

    return sorted;
}



/**
sutil::mock_merged
-------------------

**/


template<typename T>
inline NP* sutil::mock_merged(size_t ni, float time_window)
{
    NP* merged = T::zeros(ni);
    T* mm = (T*)merged->bytes();
    for(size_t i=0 ; i < ni ; i++)
    {
        T& m = mm[i];

        unsigned dummy = 1 + ( i % 10000 ) ;
        unsigned id = dummy ;
        unsigned hc = dummy ;
        float time = 100.f + float(dummy)*0.01f ;

        m.time = time ;
        m.flagmask = EFFICIENCY_COLLECT ;
        m.set_identity(id);
        m.set_hitcount(hc);
    }

    NP* merged_sorted = sort_photon_by_key<T>(merged, time_window );
    delete merged ;
    return merged_sorted ;
}




/**
sutil::unmerge
--------------

1. 1st pass, sum hitcount from merged_photon to get count of original unmerged hits
2. allocate unmerged using the count
3. 2nd pass, unfold each merged hit according to the hitcount


**/

template<typename T>
inline NP* sutil::unmerge( const NP* merged_photon )
{
    if(merged_photon == nullptr) return nullptr ;
    check_shape<T>(merged_photon);

    size_t ni = merged_photon->num_items();
    T*     pp = (T*)merged_photon->bytes();

    // 1. 1st pass, sum hitcount from merged_photon to get count of original unmerged hits
    size_t count = 0 ;
    for(size_t i=0 ; i < ni ; i++)
    {
        const T& p = pp[i];
        unsigned hc = p.hitcount();
        count += hc ;
    }

    // 2. allocate unmerged using the count

    NP* unmerged = T::zeros(count);
    T* uu = (T*)unmerged->bytes();

    // 3. 2nd pass, unfold each merged hit according to the hitcount

    size_t idx = 0 ;
    for(size_t i=0 ; i < ni ; i++)
    {
        const T& p = pp[i];
        unsigned hc = p.hitcount();

        for(size_t j=0 ; j < hc ; j++)
        {
            T& u = uu[idx];

            u = p ;  // start by duplicating everything, including time and identity
            u.set_hitcount(1);
            idx += 1 ;
        }
    }
    return unmerged ;
}


template<typename T>
inline NP* sutil::shuffle(const NP* photon, uint64_t seed )
{
    check_shape<T>(photon);
    size_t ni = photon->num_items();
    const T* src = (const T*)photon->bytes();
    NP* out = T::zeros(ni);
    T* dst = (T*)out->bytes();

    std::vector<unsigned> idx(ni);
    std::iota(idx.begin(), idx.end(), 0u);

    auto rng = seed ? std::mt19937_64(seed) : std::mt19937_64(std::random_device{}());
    std::shuffle(idx.begin(), idx.end(), rng);

    for (size_t i = 0; i < ni ; ++i) dst[i] = src[idx[i]];

    return out;
}

/**
sutil::create_photonlite_from_photon
-------------------------------------

Caution this is not localized, see stree::create_photonlite_from_photon
for the equivalent with localization using the iindex transforms
held by the stree

**/

inline void sutil::create_photonlite_from_photon( sphotonlite& lite, const sphoton& p )
{
    //lite.set_lpos( p.get_cost(), p.get_fphi() );  // NB THIS IS NOT LOCALIZED
    lite.set_lpos( 0.f, 0.f );  // NB THIS IS NOT LOCALIZED
    lite.time = p.time ;
    lite.flagmask = p.flagmask ;
    lite.set_hitcount_identity( p.hitcount(), p.get_identity() );
}


inline NP* sutil::create_photonlite_from_photon( const NP* photon )
{
    if(photon == nullptr) return nullptr ;

    size_t ni = photon->num_items();
    assert( photon->has_shape( ni, 4, 4));
    sphoton* pp = (sphoton*)photon->bytes();

    NP* photonlite = sphotonlite::zeros(ni);
    assert( photonlite->has_shape(ni, 4) );
    sphotonlite* ll = (sphotonlite*)photonlite->bytes();

    for(size_t i=0 ; i < ni ; i++)
    {
       const sphoton& p = pp[i];
       sphotonlite& l = ll[i];
       create_photonlite_from_photon( l, p );
    }
    return photonlite ;
}


