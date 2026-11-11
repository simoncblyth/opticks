#include "CSGFoundry.h"
#include "CSGFoundryCompare.h"
#include "SSim.hh"
#include "SLOG.hh"

const plog::Severity CSGFoundryCompare::LEVEL = SLOG::EnvLevel("CSGFoundryCompare", "DEBUG" );

/**
CSGFoundryCompare::Compare
---------------------------

This does very simple byte comparison : looking for equality.

TODO: find/implement absolute and relative difference comparison
using compare methods specific to the types to handle real comparisons,
such as needed by CSGCopyTest

**/

int CSGFoundryCompare::Compare( const CSGFoundry* a, const CSGFoundry* b )
{
    int mismatch = 0 ;
    mismatch += CompareVec( "solid", a->solid, b->solid );
    mismatch += CompareVec( "prim" , a->prim , b->prim );
    mismatch += CompareVec( "node" , a->node , b->node );
    mismatch += CompareVec( "plan" , a->plan , b->plan );
    mismatch += CompareVec( "tran" , a->tran , b->tran );
    mismatch += CompareVec( "itra" , a->itra , b->itra );
    mismatch += CompareVec( "inst" , a->inst , b->inst );
    mismatch += CompareVec( "gas"  , a->gas , b->gas );
    LOG_IF(fatal, mismatch != 0 ) << " mismatch FAIL ";
    //assert( mismatch == 0 );
    mismatch += SSim::Compare( a->sim, b->sim );

    return mismatch ;
}

/**
CSGFoundryCompare::WIP_CompareStruct
------------------------------

The base comparisons are not implemented yet.

**/

int CSGFoundryCompare::WIP_CompareStruct( const CSGFoundry* a, const CSGFoundry* b )
{
    int mismatch = 0 ;
    mismatch += CompareStruct( "solid", a->solid, b->solid );
    mismatch += CompareStruct( "prim" , a->prim , b->prim );
    mismatch += CompareStruct( "node" , a->node , b->node );
    mismatch += CompareFloat4( "plan" , a->plan , b->plan );
    mismatch += CompareStruct( "tran" , a->tran , b->tran );
    mismatch += CompareStruct( "itra" , a->itra , b->itra );
    mismatch += CompareStruct( "inst" , a->inst , b->inst );
    mismatch += CompareVec(    "gas"  , a->gas , b->gas );
    LOG_IF(fatal, mismatch != 0 ) << " mismatch FAIL ";
    //assert( mismatch == 0 );
    mismatch += SSim::Compare( a->sim, b->sim );

    return mismatch ;
}





std::string CSGFoundryCompare::DescCompare( const CSGFoundry* a, const CSGFoundry* b )
{
    std::stringstream ss ;
    ss << "CSGFoundryCompare::DescCompare" << std::endl ;
    int mismatch = 0 ;
    int cv = 0 ;
    cv = CompareVec( "solid", a->solid, b->solid ); mismatch += cv ;
    ss << "CompareVec.solid " <<  cv << std::endl ;
    cv = CompareVec( "prim", a->prim, b->prim );   mismatch += cv ;
    ss << "CompareVec.prim " <<  cv << std::endl ;
    cv = CompareVec( "node" , a->node , b->node );  mismatch += cv ;
    ss << "CompareVec.node " <<  cv << std::endl ;
    cv = CompareVec( "plan" , a->plan , b->plan );  mismatch += cv ;
    ss << "CompareVec.plan " <<  cv << std::endl ;
    cv = CompareVec( "tran" , a->tran , b->tran );  mismatch += cv ;
    ss << "CompareVec.tran " <<  cv << std::endl ;
    cv = CompareVec( "itra" , a->itra , b->itra );  mismatch += cv ;
    ss << "CompareVec.itra " <<  cv << std::endl ;
    cv = CompareVec( "inst" , a->inst , b->inst );  mismatch += cv ;
    ss << "CompareVec.inst " <<  cv << std::endl ;
    cv = CompareVec( "gas" , a->gas , b->gas );  mismatch += cv ;
    ss << "CompareVec.gas " <<  cv << std::endl ;
    cv = SSim::Compare( a->sim, b->sim ) ;  mismatch += cv ;
    ss << "SSim::Compare " <<  cv << std::endl ;
    ss << SSim::DescCompare( a->sim, b->sim ) << std::endl ;
    ss << " mismatch " << mismatch << std::endl ;
    std::string s = ss.str();
    return s ;
}


/**
CSGFoundryCompare::CompareVec
------------------------

Simple comparison looking for equality.

TODO: adopt svec.h

**/

template<typename T>
int CSGFoundryCompare::CompareVec( const char* name, const std::vector<T>& a, const std::vector<T>& b )
{
    int mismatch = 0 ;

    bool size_match = a.size() == b.size() ;
    LOG_IF(info, !size_match) << name << " size_match FAIL " << a.size() << " vs " << b.size()    ;
    if(!size_match) mismatch += 1 ;
    if(!size_match) return mismatch ;  // below will likely crash if sizes are different

    int data_match = memcmp( a.data(), b.data(), a.size()*sizeof(T) ) ;
    LOG_IF(info, data_match != 0) << name << " sizeof(T) " << sizeof(T) << " data_match FAIL "  ;
    if(data_match != 0) mismatch += 1 ;

    int byte_match = CompareBytes( a.data(), b.data(), a.size()*sizeof(T) ) ;
    LOG_IF(info, byte_match != 0) << name << " sizeof(T) " << sizeof(T) << " byte_match FAIL " ;
    if(byte_match != 0) mismatch += 1 ;

    LOG_IF(fatal, mismatch != 0) << " mismatch FAIL for " << name ;
    if( mismatch != 0 ) std::cout
         << " mismatch FAIL for " << name
         << " a.size " << a.size()
         << " b.size " << b.size()
         << std::endl
         ;
    return mismatch ;
}



/**
CSGFoundryCompare::CompareStruct
----------------------------

More nuanced comparison to avoid small relative differences
causing being regarded as errors.

**/


template<typename T>
int CSGFoundryCompare::CompareStruct( const char* name, const std::vector<T>& aa, const std::vector<T>& bb )
{
    int mismatch = 0 ;

    bool size_match = aa.size() == bb.size() ;
    LOG_IF(info, !size_match) << name << " size_match FAIL " << aa.size() << " vs " << bb.size()    ;
    if(!size_match) mismatch += 1 ;
    if(!size_match) return mismatch ;  // below will likely crash if sizes are different

    int num_struct = aa.size();
    int num_diff = 0 ;
    for(int i=0 ; i < num_struct ; i++)
    {
        const T& a = aa[i] ;
        const T& b = bb[i] ;
        bool is_diff = T::IsDiff( a, b );
        if( is_diff ) num_diff += 1 ;
    }
    if(num_diff != 0 ) mismatch += 1 ;

    LOG_IF(fatal, mismatch != 0) << " mismatch FAIL for " << name ;
    if( mismatch != 0 ) std::cout
         << " mismatch FAIL for " << name
         << " aa.size " << aa.size()
         << " bb.size " << bb.size()
         << " num_diff " << num_diff
         << " mismatch " << mismatch
         << std::endl
         ;
    return mismatch ;
}

int CSGFoundryCompare::CompareFloat4( const char* name, const std::vector<float4>& aa, const std::vector<float4>& bb ) // static
{
    int mismatch = 0 ;
    bool size_match = aa.size() == bb.size() ;
    LOG_IF(info, !size_match) << name << " size_match FAIL " << aa.size() << " vs " << bb.size()    ;
    if(!size_match) mismatch += 1 ;
    if(!size_match) return mismatch ;  // below will likely crash if sizes are different

    int num_struct = aa.size();
    int num_diff = 0 ;
    for(int i=0 ; i < num_struct ; i++)
    {
        const float4& a = aa[i] ;
        const float4& b = bb[i] ;
        bool is_diff = Float4_IsDiff( a, b );
        if( is_diff ) num_diff += 1 ;
    }
    if(num_diff != 0 ) mismatch += 1 ;

    LOG_IF(fatal, mismatch != 0) << " mismatch FAIL for " << name ;
    if( mismatch != 0 ) std::cout
         << " mismatch FAIL for " << name
         << " aa.size " << aa.size()
         << " bb.size " << bb.size()
         << " num_diff " << num_diff
         << " mismatch " << mismatch
         << std::endl
         ;
    return mismatch ;
}

bool CSGFoundryCompare::Float4_IsDiff( const float4& a , const float4& b ) // static
{
    return false ;
}


int CSGFoundryCompare::CompareBytes(const void* a, const void* b, unsigned num_bytes)
{
    const char* ca = (const char*)a ;
    const char* cb = (const char*)b ;
    int mismatch = 0 ;
    for(int i=0 ; i < int(num_bytes) ; i++ ) if( ca[i] != cb[i] ) mismatch += 1 ;
    return mismatch ;
}


template int CSGFoundryCompare::CompareVec(const char*, const std::vector<CSGSolid>& a, const std::vector<CSGSolid>& b ) ;
template int CSGFoundryCompare::CompareVec(const char*, const std::vector<CSGPrim>& a, const std::vector<CSGPrim>& b ) ;
template int CSGFoundryCompare::CompareVec(const char*, const std::vector<CSGNode>& a, const std::vector<CSGNode>& b ) ;
template int CSGFoundryCompare::CompareVec(const char*, const std::vector<float4>& a, const std::vector<float4>& b ) ;
template int CSGFoundryCompare::CompareVec(const char*, const std::vector<qat4>& a, const std::vector<qat4>& b ) ;
template int CSGFoundryCompare::CompareVec(const char*, const std::vector<unsigned>& a, const std::vector<unsigned>& b ) ;




