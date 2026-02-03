#pragma once
/**
s_bb.h
========

* ctor adds to the pool
* dtor removes from the pool

*/

#include <string>
#include <sstream>
#include <iomanip>

#include "s_pool.h"

struct _s_bb
{
    static constexpr const char* ITEM = "6" ;

    double x0 ;
    double y0 ;
    double z0 ;
    double x1 ;
    double y1 ;
    double z1 ;
};

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API s_bb
{
    static constexpr const int NUM = 6 ;
    static constexpr const char* NAME = "s_bb.npy" ;
    static constexpr const bool LEAK = false ;
    typedef s_pool<s_bb,_s_bb> POOL ;
    static POOL* pool ;
    static void SetPOOL( POOL* pool_ );
    static int level() ;
    //static bool IsZero( const double* v );
    static void Serialize( _s_bb& p, const s_bb* o );
    static s_bb* Import(  const _s_bb* p, const std::vector<_s_bb>& buf );

    s_bb();
    ~s_bb();
    s_bb* copy() const ;

    int pid ;

    double x0 ;
    double y0 ;
    double z0 ;
    double x1 ;
    double y1 ;
    double z1 ;


    const double* cdata() const { return &x0 ; }
    double* data() {              return &x0 ; }

    bool is_root() const { return true ; }
    std::string desc() const ;
    template<typename T> void write( T* dst ) const ;

    template<typename T> static std::string Desc( const T* aabb );
    template<typename T> static bool AllZero( const T* aabb );
    template<typename T> static bool Degenerate( const T* aabb );

    template<typename T, int N> static std::string Desc_( const T* v );
    template<typename T, int N> static std::string DescNumPy_( const T* v, const char* symbol = "bb", bool imp = false );
    template<typename T, int N> static bool AllZero_( const T* v );
    template<typename T, int N> static bool Degenerate_( const T* v );

    template<typename S, typename T>
    static void IncludePoint( T* aabb,  const S* other_point );

    template<typename S, typename T>
    static void IncludeAABB(  T* aabb,  const S* other_aabb , std::ostream* out=nullptr  );

    template<typename T> static T    Extent( const T* aabb ) ;
    template<typename T> static T    AbsMax( const T* aabb ) ;
    template<typename T> static void CenterExtent( T* ce,  const T* aabb ) ;

    void include_point(      const double* point );
    void include_point_widen( const float*  point );
    void include_aabb(       const double* aabb  );
    void include_aabb_widen(  const float* aabb  );

    template<typename T>
    void center_extent( T* ce ) const ;

    template<typename T>
    static bool HasOverlap( const T* a,  const T* b ) ;

};

inline void s_bb::SetPOOL( POOL* pool_ ){ pool = pool_ ; }
inline int s_bb::level() {  return pool ? pool->level : ssys::getenvint("sn__level",-1) ; } // static


inline void s_bb::Serialize( _s_bb& p, const s_bb* o )
{
    p.x0 = o->x0 ;
    p.y0 = o->y0 ;
    p.z0 = o->z0 ;
    p.x1 = o->x1 ;
    p.y1 = o->y1 ;
    p.z1 = o->z1 ;
}
inline s_bb* s_bb::Import( const _s_bb* p, const std::vector<_s_bb>& )
{
    s_bb* o = new s_bb ;
    o->x0 = p->x0 ;
    o->y0 = p->y0 ;
    o->z0 = p->z0 ;
    o->x1 = p->x1 ;
    o->y1 = p->y1 ;
    o->z1 = p->z1 ;
    return o ;
}


inline s_bb::s_bb()
    :
    pid(pool ? pool->add(this) : -1),
    x0(0.),
    y0(0.),
    z0(0.),
    x1(0.),
    y1(0.),
    z1(0.)
{
    if(level() > 1) std::cerr << "s_bb::s_bb pid " << pid << std::endl ;
}
inline s_bb::~s_bb()
{
    if(level() > 1) std::cerr << "s_bb::~s_bb pid " << pid << std::endl ;
    if(pool) pool->remove(this);
}

/**
s_bb::copy
-----------

Note that *pid* is not copied, as pid is set by s_pool::add within the ctor.
This is because pid is only relevant to the persisting architecture
rather than being part of the payload.

**/

inline s_bb* s_bb::copy() const
{
    s_bb* n = new s_bb ;
    n->x0 = x0 ;
    n->y0 = y0 ;
    n->z0 = z0 ;
    n->x1 = x1 ;
    n->y1 = y1 ;
    n->z1 = z1 ;
    return n ;
}

inline std::string s_bb::desc() const { return Desc(cdata()) ; }

template<typename T>
inline void s_bb::write( T* dst ) const
{
    const double* src = cdata();
    for(int i=0 ; i < NUM ; i++) dst[i] = T(src[i]) ;
}



template<typename T>
inline std::string s_bb::Desc( const T* aabb ) // static
{
    return Desc_<T,6>(aabb);
}
template<typename T, int N>
inline std::string s_bb::Desc_( const T* v ) // static
{
    int w = 10 ;
    int p = 3 ;
    bool all_zero = AllZero_<T,N>(v);
    bool degenerate = Degenerate_<T,N>(v);
    std::stringstream ss ;
    ss << "[" ;
    for(int i=0 ; i < N ; i++)
        ss << std::fixed << std::setw(w) << std::setprecision(p) << v[i] << ( i < N - 1 ? "," : "" )  ;
    ss << "]"
       << ( all_zero ? " ALL_ZERO" : "" )
       << ( degenerate ? " DEGENERATE" : "" )
       ;
    std::string str = ss.str();
    return str ;
}

template<typename T, int N>
inline std::string s_bb::DescNumPy_( const T* v, const char* symbol, bool imp ) // static
{
    int w = 10 ;
    int p = 3 ;

    std::stringstream ss ;
    if(imp) ss << "import numpy as np " ;
    ss << " ; " << symbol << " = np.array([" ;
    for(int i=0 ; i < N ; i++)
        ss << std::fixed << std::setw(w) << std::setprecision(p) << v[i] << ( i < N - 1 ? "," : "" )  ;
    ss << "]) " ;
    std::string str = ss.str();
    return str ;
}





template<typename T>
inline bool s_bb::AllZero( const T* v ) // static
{
    return AllZero_<T,6>(v) ;
}
template<typename T, int N>
inline bool s_bb::AllZero_( const T* v ) // static
{
    int count = 0 ;
    for(int i=0 ; i < N ; i++) if(std::abs(v[i]) == T(0.)) count += 1 ;
    return count == N ;
}



template<typename T>
inline bool s_bb::Degenerate( const T* v ) // static
{
    return Degenerate_<T,6>(v) ;
}
template<typename T, int N>
inline bool s_bb::Degenerate_( const T* v ) // static
{
    int count = 0 ;
    for(int i=0 ; i < N/2 ; i++) if(v[i] == v[i+N/2]) count += 1 ;
    return count == N/2 ;
}





/**
s_bb::IncludePoint
-------------------

When aabb starts empty the point is adopted as both min and max
without any comparisons.

HMM: AN ALL ZERO POINT IS TREATED AS VALID

**/

template<typename S,typename T>
inline void s_bb::IncludePoint( T* aabb,  const S* point ) // static
{
    bool adopt = AllZero(aabb) ;
    assert( NUM == 6 );
    for(int i=0 ; i < 3   ; i++) aabb[i] = adopt ? T(point[i])   : std::min( aabb[i], T(point[i])   );
    for(int i=3 ; i < NUM ; i++) aabb[i] = adopt ? T(point[i-3]) : std::min( aabb[i], T(point[i-3]) );
}



/**
s_bb::IncludeAABB
------------------

When aabb starts empty and the other_aabb is not empty the
other_aabb is adopted with no comparisons.
Otherwise when both have values the min and max of the
aabb gets set by comparison.

NB all zero other_aabb values causes an early exit and no combination
as such an other_aabb is regarded as unset

**/

template<typename S,typename T>
inline void s_bb::IncludeAABB(  T* aabb,  const S* other_aabb, std::ostream* out  ) // static
{
    bool other_aabb_zero = AllZero(other_aabb) ;
    if(other_aabb_zero) return ;

    bool aabb_zero = AllZero(aabb) ;

    if(out) *out
        << "s_bb::IncludeAABB "
        << std::endl
        << " inital_aabb  " << Desc(aabb)
        << std::endl
        << " other_aabb   " << Desc(other_aabb)  << ( aabb_zero ? " ADOPT OTHER AS STARTING" : "COMBINE" )
        << std::endl
        ;

    assert( NUM == 6 );
    for(int i=0 ; i < 3   ; i++) aabb[i] = aabb_zero ? T(other_aabb[i]) : std::min(aabb[i], T(other_aabb[i]) ) ;
    for(int i=3 ; i < NUM ; i++) aabb[i] = aabb_zero ? T(other_aabb[i]) : std::max(aabb[i], T(other_aabb[i]) ) ;

    if(out) *out
        << " updated_aabb " << Desc(aabb)
        << std::endl
        ;
}












template<typename T>
inline T s_bb::Extent( const T* aabb )  // static
{
    return std::max(std::max(aabb[3+0]-aabb[0],aabb[3+1]-aabb[1]),aabb[3+2]-aabb[2])/T(2.) ;
}

template<typename T>
inline T s_bb::AbsMax( const T* aabb )  // static
{
    T mx = 0. ;
    for(int i=0 ; i < 6 ; i++)
    {
        T amx = std::abs(aabb[i]);
        if( amx > mx ) mx = amx ;
    }
    return mx ;
}



template<typename T>
inline void s_bb::CenterExtent( T* ce,  const T* aabb )  // static
{
    for(int i=0 ; i < 3 ; i++) ce[i] = (aabb[i] + aabb[i+3])/T(2.) ;
    ce[3] = Extent(aabb) ;
}



inline void s_bb::include_point( const double* point )
{
    IncludePoint<double,double>( data(), point );
}
inline void s_bb::include_point_widen( const float* point )
{
    IncludePoint<float,double>( data(), point );
}


inline void s_bb::include_aabb(  const double* aabb  )
{
    IncludeAABB<double,double>( data(), aabb );
}
/**

s_bb::include_aabb_widen
--------------------------

Widening is needed because CSGNode/CSGPrim are float based.
Keeping those in double and only narrowing just before the
upload is a possibility.  That is not totally trivial however
because they carry integers in some of the elements.

**/

inline void s_bb::include_aabb_widen(  const float* aabb  )
{
    IncludeAABB<float,double>( data(), aabb );
}

template<typename T>
inline void s_bb::center_extent( T* ce ) const
{
    ce[0] = ( x0 + x1 )/2. ;
    ce[1] = ( y0 + y1 )/2. ;
    ce[2] = ( z0 + z1 )/2. ;
    ce[3] = std::max(std::max(x1-x0,y1-y0),z1-z0)/2. ;  // 2026-2-3 fix omitted /2.
    assert(0); // checking users
}


template<typename T>
inline bool s_bb::HasOverlap( const T* a,  const T* b )
{
    return (a[0] <= b[3] && a[3] >= b[0]) && // X overlap
           (a[1] <= b[4] && a[4] >= b[1]) && // Y overlap
           (a[2] <= b[5] && a[5] >= b[2]);   // Z overlap
}






