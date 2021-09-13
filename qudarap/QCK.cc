
#include <iostream>
#include <iomanip>
#include "NP.hh"
#include "QCK.hh"

void QCK::save(const char* base, const char* reldir) const 
{
    rindex->save(base, reldir, "rindex.npy"); 
    bis->save(base, reldir, "bis.npy"); 
    s2c->save(base, reldir, "s2c.npy"); 
    s2cn->save(base, reldir, "s2cn.npy"); 
}

QCK* QCK::Load(const char* base, const char* reldir)   // static
{
    NP* rindex = NP::Load(base, reldir, "rindex.npy"); 
    NP* bis = NP::Load(base, reldir, "bis.npy"); 
    NP* s2c = NP::Load(base, reldir, "s2c.npy"); 
    NP* s2cn = NP::Load(base, reldir, "s2cn.npy"); 

    QCK* qck = new QCK ; 

    qck->rindex = rindex ; 
    qck->bis = bis ; 
    qck->s2c = s2c ; 
    qck->s2cn = s2cn ; 

    return qck ; 
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

template<typename T> T QCK::energy_lookup( const T BetaInverse, const T u) const 
{
    bool in_range ; 
    unsigned column = 0u ; // placeholder, as bis is 1d
    int ibin = bis->pfindbin<T>(BetaInverse, column, in_range ); 
    assert( in_range == true ); 

    int item = ibin - 1 ;  // ibin is 1-based for in_range bins, so item is 0-based  
    assert( item > -1 );  

    bool dump = false ;  
    T en = s2cn->pdomain<T>(u, item, dump );

/*
    int w = 10 ; 
    int p = 6 ; 
    std::cout  
        << "QCK::energy_lookup"  
        << " BetaInverse "      << std::fixed << std::setw(w) << std::setprecision(p) << BetaInverse 
        << " in_range "         << std::setw(5) << in_range 
        << " ibin "             << std::setw(5) << ibin
        << " item "             << std::setw(5) << item
        << " u  "               << std::fixed << std::setw(w) << std::setprecision(p) << u 
        << " en  "              << std::fixed << std::setw(w) << std::setprecision(p) << en 
        << std::endl
        ;
*/

    return en  ; 
}
template float  QCK::energy_lookup( const float, const float ) const ; 
template double QCK::energy_lookup( const double, const double ) const ; 




template<typename T> NP* QCK::energy_lookup( const T BetaInverse, const NP* uu) const 
{
    unsigned ndim = uu->shape.size() ; 
    assert( ndim == 1 ); 
    unsigned ni = uu->shape[0] ; 
    const T* vv = uu->cvalues<T>(); 

    NP* en = NP::MakeLike(uu); 
    T* ee = en->values<T>(); 
    for(unsigned i=0 ; i < ni ; i++) ee[i] = energy_lookup( BetaInverse, vv[i] ) ; 

    return en ; 
}
template NP* QCK::energy_lookup( const float, const NP* uu ) const ; 
template NP* QCK::energy_lookup( const double, const NP* uu ) const ; 




template<typename T> bool QCK::is_permissable( const T BetaInverse) const 
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
        << " bis.get_edges.lo " << std::fixed << std::setw(w) << std::setprecision(p) << lo 
        << " bis.get_edges.hi " << std::fixed << std::setw(w) << std::setprecision(p) << hi 
        << std::endl
        ;
         
    assert( BetaInverse >= lo && BetaInverse <= hi ); 
    return in_range ; 
}
template bool QCK::is_permissable( const float ) const ; 
template bool QCK::is_permissable( const double ) const ; 




