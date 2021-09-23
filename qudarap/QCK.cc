
#include <iostream>
#include <iomanip>
#include <chrono>

#include "NP.hh"
#include "QCK.hh"
#include "PLOG.hh"

template<typename T>
void QCK<T>::save(const char* base, const char* reldir) const 
{
    rindex->save(base, reldir, "rindex.npy"); 
    bis->save(base, reldir, "bis.npy"); 
    avph->save(base, reldir, "avph.npy"); 
    s2c->save(base, reldir, "s2c.npy"); 
    s2cn->save(base, reldir, "s2cn.npy"); 
}

template<typename T>
QCK<T>* QCK<T>::Load(const char* base, const char* reldir)   // static
{
    NP* rindex = NP::Load(base, reldir, "rindex.npy"); 
    NP* bis = NP::Load(base, reldir, "bis.npy"); 
    NP* avph = NP::Load(base, reldir, "avph.npy"); 
    NP* s2c = NP::Load(base, reldir, "s2c.npy"); 
    NP* s2cn = NP::Load(base, reldir, "s2cn.npy"); 

    QCK* qck = new QCK<T> ; 

    qck->rindex = rindex ; 
    qck->bis = bis ; 
    qck->avph = avph ; 
    qck->s2c = s2c ; 
    qck->s2cn = s2cn ; 

    if(rindex == nullptr || bis == nullptr || s2c == nullptr || s2cn == nullptr || avph == nullptr)
    {
        LOG(fatal)
            << " QCK::Load incomplete "
            << " base " << base
            << " reldir " << reldir
            << std::endl 
            << " rindex " << rindex 
            << " bis " << bis
            << " s2c " << s2c
            << " s2cn " << s2cn
            << std::endl 
            << " use QCerenkovTest:test_makeICDF to recreate these arrays "
            ;
    }
    return qck ; 
}

template<typename T> 
void QCK<T>::init()
{

    assert( rindex ) ; 
    assert( rindex->shape.size() == 2 ); 
    rindex->minmax<T>(emn, emx, 0u ); 
    rindex->minmax<T>(rmn, rmx, 1u ); 

    int w = 10 ; 
    int p = 5 ; 

    LOG(info)  
        << " emn " << std::setw(w) << std::fixed << std::setprecision(p) << emn 
        << " emx " << std::setw(w) << std::fixed << std::setprecision(p) << emx 
        << " rmn " << std::setw(w) << std::fixed << std::setprecision(p) << rmn 
        << " rmx " << std::setw(w) << std::fixed << std::setprecision(p) << rmx 
        ;


}


template<typename T> int QCK<T>::find_bisIdx( const T BetaInverse ) const 
{
    bool in_range ; 
    unsigned column = 0u ; // placeholder, as bis is 1d
    int ibin = bis->pfindbin<T>(BetaInverse, column, in_range ); 
    assert( in_range == true ); 
    int item = ibin - 1 ;  // ibin is 1-based for in_range bins, so item is 0-based  
    assert( item > -1 );  

/*
    int w = 10 ; 
    int p = 6 ; 
    std::cout  
        << "QCK::find_bisIdx"  
        << " BetaInverse "      << std::fixed << std::setw(w) << std::setprecision(p) << BetaInverse 
        << " in_range "         << std::setw(5) << in_range 
        << " ibin "             << std::setw(5) << ibin
        << " item "             << std::setw(5) << item
        << std::endl
        ;
*/

    return item ; 
}



/*
QCK::energy_lookup
----------------------

For BetaInverse such as 1.0 which is less than RINDEX_min 
Cerenkov can happen anywhere across the full energy range.

For BetaInverse approaching the limits of where Cerenkov
can happen, like when there is only a small RINDEX island
peaking above the BetaInverse sea the ICDF will be shaped
more and more like a step function, so no matter the 
input random number will get close to the same energy.

For BetaInverse that is greater than RINDEX_max the "island" has 
been submerged and Cerenkov does not happen, **so this should not be called**
as it will trip an assert.

Avoid tripping the assert by only calling QCK::energy_looking with BetaInverse 
that returns true from QCK::is_permissable. 
 
see ~/np/tests/NPget_edgesTest.cc

*/

template<typename T> T QCK<T>::energy_lookup_( const T BetaInverse, const T u, double& dt ) const 
{
    typedef std::chrono::high_resolution_clock::time_point TP ;
    TP t0 = std::chrono::high_resolution_clock::now() ;

    int ibis = find_bisIdx(BetaInverse); 

    bool dump = false ;  
    T en = s2cn->pdomain<T>(u, ibis, dump );

    TP t1 = std::chrono::high_resolution_clock::now() ; 
    std::chrono::duration<double> t10 = t1 - t0; 
    dt = t10.count()*DT_SCALE ;        

    return en  ; 
}

/**
QCK::energy_range_s2cn
------------------------

For _SplitBin the energy range obtained from s2cn is the full energy range.

**/

template<typename T> void QCK<T>::energy_range_s2cn( T& emin, T& emax, const T BetaInverse, bool dump ) const 
{
    int ibis = find_bisIdx(BetaInverse); 
    unsigned k = 0 ; // 1st payload slot is domain, the energy 
    emin = s2cn->get<T>(ibis,  0, k) ;   // first energy 
    emax = s2cn->get<T>(ibis, -1, k) ;   // last energy 

    T edom = emax - emin ; 
    if(dump) std::cout 
        << "QCK::energy_range_s2cn" 
        << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse
        << " ibis " << std::setw(10) << ibis
        << " bis[ibis] " <<  std::setw(10) << std::fixed << std::setprecision(4) << bis->get<T>(ibis) 
        << " emin " << std::setw(10) << std::fixed << std::setprecision(4) << emin 
        << " emax " << std::setw(10) << std::fixed << std::setprecision(4) << emax 
        << " edom " << std::setw(10) << std::fixed << std::setprecision(4) << edom 
        << std::endl 
        ;
}

template<typename T> void QCK<T>::energy_range_avph( T& emin, T& emax, const T BetaInverse, bool dump ) const 
{
    int ibis = find_bisIdx(BetaInverse); 
    T BetaInverse2 = avph->get<T>(ibis,0) ; 
    emin = avph->get<T>(ibis,1) ;  
    emax = avph->get<T>(ibis,2) ; 
    T averageNumPhotons = avph->get<T>(ibis,3) ; 

    T edom = emax - emin ; 
    if(dump) std::cout 
        << "QCK::energy_range_avph" 
        << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse
        << " BetaInverse2 " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse2
        << " ibis " << std::setw(10) << ibis
        << " bis[ibis] " <<  std::setw(10) << std::fixed << std::setprecision(4) << bis->get<T>(ibis) 
        << " emin " << std::setw(10) << std::fixed << std::setprecision(4) << emin 
        << " emax " << std::setw(10) << std::fixed << std::setprecision(4) << emax 
        << " edom " << std::setw(10) << std::fixed << std::setprecision(4) << edom 
        << " averageNumPhotons " << std::setw(10) << std::fixed << std::setprecision(4) << averageNumPhotons
        << std::endl 
        ;
}



template<typename T> const double QCK<T>::DT_SCALE = 1e6 ; 

/**
QCK::energy_sample_
----------------------

*u1_s2max* needs to be less than s2 for the energy to be accepted 

Using full emn to emx range is problematic for BetaInverse close to the 
RINDEX peak as most of the randoms will be wasted because the vast majority 
of the range is not Cerenkov permitted. 


**/

template<typename T> T QCK<T>::energy_sample_( const T emin, const T emax, const T BetaInverse, const std::function<T()>& rng, double& dt, unsigned& count ) const 
{
    typedef std::chrono::high_resolution_clock::time_point TP ;
    TP t0 = std::chrono::high_resolution_clock::now() ;

    const T one(1.) ; 
    T u0, u1, en, ri, ct, s2, u1_s2max ; 

    const T ctmax = BetaInverse/rmx ; 
    const T s2max = ( one - ctmax )*( one + ctmax ) ; 
    count = 0 ; 

    do {
        u0 = rng() ; 
        u1 = rng() ; 

        en = emin + u0*(emax-emin) ; 
        ri = rindex->interp<T>(en) ;   
        ct = BetaInverse/ri ; 
        s2 = ( one - ct )*( one + ct ) ;
        u1_s2max = u1*s2max ;  

        count += 1 ; 

    } while ( u1_s2max > s2 ) ; 


    TP t1 = std::chrono::high_resolution_clock::now() ; 
    std::chrono::duration<double> t10 = t1 - t0; 
    dt = t10.count()*DT_SCALE ;        

    return en ; 
}


template<typename T> NP* QCK<T>::energy_lookup( const T BetaInverse, const NP* uu, NP* tt ) const 
{
    unsigned ndim = uu->shape.size() ; 
    assert( ndim == 1 ); 
    unsigned ni = uu->shape[0] ; 
    const T* uu_v = uu->cvalues<T>(); 

    if(tt) assert( tt->shape.size() == 1 && tt->shape[0] == ni ); 
    T* tt_v = tt ? tt->values<T>() : nullptr ; 


    NP* en = NP::MakeLike(uu); 
    T* en_v = en->values<T>(); 
    double dt ; 

    for(unsigned i=0 ; i < ni ; i++) 
    {
        en_v[i] = energy_lookup_( BetaInverse, uu_v[i],  dt ) ; 
        if(tt_v) tt_v[i] = dt ; 
    }
    return en ; 
}

/**
QCK::energy_sample
-------------------

Returns array of energy samples of shape (ni,) and if argument tt is non-null and of shape (ni,) sets cputimes

**/

template<typename T> NP* QCK<T>::energy_sample( const T BetaInverse, const std::function<T()>& rng, unsigned ni, NP* tt ) const 
{
    NP* en = NP::Make<T>(ni); 
    T* en_v = en->values<T>(); 

    if(tt) assert( tt->shape.size() == 1 && tt->shape[0] == ni ); 
    T* tt_v = tt ? tt->values<T>() : nullptr ; 

    double dt ; 
    unsigned count_max(0); 
    unsigned count(0) ; 

    T emin ; 
    T emax ; 
    bool dump = false ; 
    energy_range_avph( emin, emax, BetaInverse, dump); 


    for(unsigned i=0 ; i < ni ; i++) 
    {
        en_v[i] = energy_sample_( emin, emax, BetaInverse, rng, dt, count  ) ; 
        if(tt_v) tt_v[i] = dt ; 

        if( count > count_max ) count_max = count ; 

        if(count > 1000) 
            std::cout 
                << "QCK::energy_sample " 
                << " i " << std::setw(7) << i 
                << " count " << std::setw(7) << count
                << std::endl 
                ; 

    }

    LOG(info) << " count_max " << count_max ; 
    return en ; 
}



/**
QCK::is_permissable
----------------------

Look for the BetaInverse value within the bis array, returning true when in range.  

**/

template<typename T> bool QCK<T>::is_permissable( const T BetaInverse) const 
{
    assert( bis->shape.size() == 1 ); 

    bool in_range ; 
    unsigned column = 0u ; // placeholder, as bis is 1d
    int ibin = bis->pfindbin<T>(BetaInverse, column, in_range ); 

    T lo, hi ; 
    bis->get_edges<T>(lo,hi,column,ibin);    // when not in_range the  hi and lo are edge values 

    int w = 10 ; 
    int p = 6 ; 

    std::cout  
        << "QCK::is_permissable "  
        << " BetaInverse "      << std::fixed << std::setw(w) << std::setprecision(p) << BetaInverse 
        << " bis.ibin "         << std::setw(5) << ibin
        << " bis.in_range "     << std::setw(5) << in_range 
        << " bis.get_edges [" 
        << " " << std::fixed << std::setw(w) << std::setprecision(p) << lo 
        << " " << std::fixed << std::setw(w) << std::setprecision(p) << hi 
        << "]"
        << std::endl
        ;
         
    assert( BetaInverse >= lo && BetaInverse <= hi ); 
    return in_range ; 
}


template struct QCK<float> ; 
template struct QCK<double> ; 
