#pragma once
/**
stran.h : Tran
================

* FOR CLARITY AM MOVING SINGLE TRANFORM FUNCTIONALITY TO stra.h 

Transform handler that creates inverse transforms at every stage  
loosely based on NPY nmat4triple_

Aim to avoid having to take the inverse eg with glm::inverse 
which inevitably risks numerical issues.

**/

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <array>
#include <vector>
#include <csignal>
#include <cstdlib>

#include "sqat4.h"
#include "stra.h"
#include "NP.hh"

#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>



template<typename D, typename S>
void TranConvert(glm::tmat4x4<D>& dst, const glm::tmat4x4<S>& src )  // static
{
    const S* src_ptr = glm::value_ptr(src); 
    D* dst_ptr = glm::value_ptr(dst); 
    for(int i=0 ; i < 16 ; i++) dst_ptr[i] = D(src_ptr[i]) ; 
}



template<typename T>
struct Tran
{
    glm::tmat4x4<T> t ;  // transform 
    glm::tmat4x4<T> v ;  // inverse  
    glm::tmat4x4<T> i ;  // identity


    static constexpr const T EPSILON = 1e-6 ; 
    static constexpr const bool VERBOSE = false ;  

    static const Tran<T>* make_translate( const T tx, const T ty, const T tz, const T sc);
    static const Tran<T>* make_translate( const T tx, const T ty, const T tz);
    static const Tran<T>* make_identity();
    static       Tran<T>* make_identity_();
    static const Tran<T>* make_scale(     const T sx, const T sy, const T sz);
    static const Tran<T>* make_rotate(    const T ax, const T ay, const T az, const T angle_deg);

    static const Tran<T>* make_rotate_a2b( const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip ); 
    static const Tran<T>* make_rotate_a2b( const T ax, const T ay, const T az, const T bx, const T by, const T bz, bool flip ); 

    static const Tran<T>* product(const Tran<T>* a, const Tran<T>* b, bool reverse);
    static const Tran<T>* product(const Tran<T>* a, const Tran<T>* b, const Tran<T>* c, bool reverse);
    static const Tran<T>* product(const std::vector<const Tran<T>*>& tt, bool reverse );

    static Tran<T>* ConvertToTran( const qat4* q, T epsilon=EPSILON ); 
    static Tran<T>* ConvertFromQat(const qat4* q_, T epsilon=EPSILON ); 

    static Tran<T>* ConvertFromData(const T* data); 

    static const qat4* Invert( const qat4* q, T epsilon=EPSILON ); 
    static Tran<T>* FromPair( const qat4* t, const qat4* v, T epsilon=EPSILON ); // WIDENS from float  
    static glm::tmat4x4<T> MatFromQat( const qat4* q );
    static glm::tmat4x4<T> MatFromData(const T* data );

    static qat4*    ConvertFrom(const glm::tmat4x4<T>& tr ); 

    // ctor
    Tran( const T* transform, const T* inverse ) ;
    Tran( const glm::tmat4x4<T>& transform, const glm::tmat4x4<T>& inverse ) ;

    T    maxdiff_from_identity(char mat='t') const ; 
    bool is_identity(char mat='t', T epsilon=1e-6) const ; 
    std::string brief(bool only_tlate=false, char mat='t', unsigned wid=6, unsigned prec=1) const ;  
    std::string desc() const ; 
    bool checkIsIdentity(char mat='i', const char* caller="caller", T epsilon=EPSILON); 
 
    void write(T* dst, unsigned num_values=3*4*4) const ; 


    NP* as_array()const ; 
    void save_(const char* path) const ; 
    void save(const char* dir, const char* name="stran.npy") const ;

    static void Apply(        const glm::tmat4x4<T>& tr, T* p0    , T w    , unsigned count, unsigned stride, unsigned offset, bool normalize ); 
    static void ApplyToFloat( const glm::tmat4x4<T>& tr, float* p0, float w, unsigned count, unsigned stride, unsigned offset, bool normalize ); 
 
    void photon_transform( NP* ph, bool normalize ) const ; 

    static NP* PhotonTransform( const NP* ph, bool normalize, const Tran<T>* tr ); 
    static void AddTransform( T* ttk, const char* opt, const glm::tvec3<T>& a, const glm::tvec3<T>& b, const glm::tvec3<T>& c ); 

    const T* tdata() const ; 
    const T* vdata() const ; 
 
};



template<typename T>
inline std::ostream& operator<< (std::ostream& out, const glm::tmat4x4<T>& m  )
{
    int prec = 4 ;   
    int wid = 10 ; 
    bool flip = false ; 
    for(int i=0 ; i < 4 ; i++)
    {   
        for(int j=0 ; j < 4 ; j++) out << std::setprecision(prec) << std::fixed << std::setw(wid) << ( flip ? m[j][i] : m[i][j] ) << " " ; 
        out << std::endl ; 
    }   
    return out ; 
}

template<typename T>
inline std::ostream& operator<< (std::ostream& out, const Tran<T>& tr)
{
    out 
       << std::endl 
       << "tr.t" 
       << std::endl 
       <<  tr.t 
       << std::endl 
       << "tr.v" 
       << std::endl 
       <<  tr.v  
       << "tr.i" 
       << std::endl 
       <<  tr.i  
       << std::endl 
       ;   
    return out;
}


template<typename T>
inline const Tran<T>* Tran<T>::make_translate( const T tx, const T ty, const T tz, const T sc)
{
    glm::tvec3<T> tlate(tx*sc,ty*sc,tz*sc); 
    glm::tmat4x4<T> t = glm::translate(glm::tmat4x4<T>(1.),   tlate ) ;
    glm::tmat4x4<T> v = glm::translate(glm::tmat4x4<T>(1.),  -tlate ) ;
    return new Tran<T>(t, v);    
}
template<typename T>
inline const Tran<T>* Tran<T>::make_translate( const T tx, const T ty, const T tz)
{
    glm::tvec3<T> tlate(tx,ty,tz); 
    glm::tmat4x4<T> t = glm::translate(glm::tmat4x4<T>(1.),   tlate ) ;
    glm::tmat4x4<T> v = glm::translate(glm::tmat4x4<T>(1.),  -tlate ) ;
    return new Tran<T>(t, v);    
}

template<typename T>
inline const Tran<T>* Tran<T>::make_identity()
{
    glm::tmat4x4<T> t(1.) ;
    glm::tmat4x4<T> v(1.) ; 
    return new Tran<T>(t, v);    
}
template<typename T>
inline Tran<T>* Tran<T>::make_identity_()
{
    glm::tmat4x4<T> t(1.) ;
    glm::tmat4x4<T> v(1.) ; 
    return new Tran<T>(t, v);    
}


template<typename T>
inline const Tran<T>* Tran<T>::make_scale( const T sx, const T sy, const T sz)
{
    glm::tvec3<T> scal(sx,sy,sz); 
    glm::tvec3<T> isca(1./sx,1./sy,1./sz); 
    glm::tmat4x4<T> t = glm::scale(glm::tmat4x4<T>(1.),   scal ) ;
    glm::tmat4x4<T> v = glm::scale(glm::tmat4x4<T>(1.),   isca ) ;
    return new Tran<T>(t, v);    
}

template<typename T>
inline const Tran<T>* Tran<T>::make_rotate( const T ax, const T ay, const T az, const T angle_deg)
{
    T angle_rad = glm::pi<T>()*angle_deg/T(180.) ;
    glm::tvec3<T> axis(ax,ay,az); 
    glm::tmat4x4<T> t = glm::rotate(glm::tmat4x4<T>(1.),  angle_rad, axis ) ;
    glm::tmat4x4<T> v = glm::rotate(glm::tmat4x4<T>(1.), -angle_rad, axis ) ;
    return new Tran<T>(t, v);    
}


template<typename T>
inline const Tran<T>* Tran<T>::make_rotate_a2b(const T ax, const T ay, const T az, const T bx, const T by, const T bz, bool flip )
{
    glm::tvec3<T> a(ax,ay,az); 
    glm::tvec3<T> b(bx,by,bz); 
    return make_rotate_a2b(a,b, flip); 
}

template<typename T>
inline const Tran<T>* Tran<T>::make_rotate_a2b(const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip )
{
    glm::tmat4x4<T> t = stra<T>::RotateA2B(a,b,flip) ; 
    glm::tmat4x4<T> v = stra<T>::RotateA2B(b,a,flip) ; 
    return new Tran<T>(t, v);    
}






template<typename T>
inline const Tran<T>* Tran<T>::product(const Tran<T>* a, const Tran<T>* b, bool reverse)
{
    std::vector<const Tran<T>*> tt ; 
    tt.push_back(a);
    tt.push_back(b);
    return Tran<T>::product( tt, reverse );
}

template<typename T>
inline const Tran<T>* Tran<T>::product(const Tran<T>* a, const Tran<T>* b, const Tran<T>* c, bool reverse)
{
    std::vector<const Tran<T>*> tt ; 
    tt.push_back(a);
    tt.push_back(b);
    tt.push_back(c);
    return Tran<T>::product( tt, reverse );
}





/**
Tran::product
--------------

Tran houses transforms paired with their inverse transforms, the product 
of the transforms and opposite order product of the inverse transforms
is done using inclusive indices to access Tran from the vector::

   i: 0 -> ntt - 1      ascending 
   j: ntt - 1 -> 0      descending (from last transform down to first)

Use *reverse=true* when the transforms are in reverse heirarchical order, ie when
they have been collected by starting from the leaf node and then following parent 
links back up to the root node. 

When combining s:scale r:rotation and t:translate the typical ordering is s-r-t 
because wish to scale and orient about a nearby (local frame) origin before 
translating into position.  


**/
template<typename T>
inline const Tran<T>* Tran<T>::product(const std::vector<const Tran<T>*>& tt, bool reverse )
{
    unsigned ntt = tt.size();
    if(ntt==0) return NULL ; 
    if(ntt==1) return tt[0] ;

    glm::tmat4x4<T> t(T(1)) ;
    glm::tmat4x4<T> v(T(1)) ;

    for(unsigned i=0,j=ntt-1 ; i < ntt ; i++,j-- )
    {
        const Tran<T>* ii = tt[reverse ? j : i] ;  // reverse:true starts from the last (ie root node)
        const Tran<T>* jj = tt[reverse ? i : j] ;  // reverse:true starts from the first (ie leaf node)

        if( ii != nullptr )  t *= ii->t ;  
        if( jj != nullptr )  v *= jj->v ;  // inverse-transform product in opposite order
    }
    return new Tran<T>(t, v) ;
}





template<typename T>
inline Tran<T>::Tran( const T* transform, const T* inverse ) 
    :   
    t(glm::make_mat4x4<T>(transform)), 
    v(glm::make_mat4x4<T>(inverse)),
    i(t*v)
{
} 

template<typename T>
inline Tran<T>::Tran( const glm::tmat4x4<T>& transform, const glm::tmat4x4<T>& inverse ) 
    :   
    t(transform), 
    v(inverse),
    i(transform*inverse)
{
} 


template<typename T>
inline const T*  Tran<T>::tdata() const 
{
    return glm::value_ptr(t) ; 
}

template<typename T>
inline const T*  Tran<T>::vdata() const 
{
    return glm::value_ptr(v) ; 
}


template<typename T>
inline T Tran<T>::maxdiff_from_identity(char mat) const 
{
    const glm::tmat4x4<T>& m = mat == 't' ? t : ( mat == 'v' ? v : i ) ; 
    T mxdif = 0. ; 
    for(int j=0 ; j < 4 ; j++ ) 
    for(int k=0 ; k < 4 ; k++ ) 
    {
        T val = m[j][k] ; 
        T xval = j == k ? T(1) : T(0) ; 
        T dif = std::abs( val - xval ) ; 
        if(dif > mxdif) mxdif = dif ; 
    }
    return mxdif ; 
}

template<typename T>
inline bool Tran<T>::is_identity(char mat, T epsilon) const 
{
    T mxdif = maxdiff_from_identity(mat) ; 
    return mxdif < epsilon ; 
}


template<typename T>
inline std::string Tran<T>::brief(bool only_tlate, char mat, unsigned wid, unsigned prec) const 
{
    std::stringstream ss ; 
    ss << mat << ":" ; 
    if(is_identity(mat)) 
    {
       ss << "identity" ; 
    }
    else
    {
        const glm::tmat4x4<T>& m = mat == 't' ? t : ( mat == 'v' ? v : i ) ; 
        int j0 = only_tlate ? 3 : 0 ; 
        for(int j=j0 ; j < 4 ; j++ ) 
        {
            ss << "[" ;
            for(int k=0 ; k < 4 ; k++ ) ss << std::setw(wid) << std::fixed << std::setprecision(prec) << m[j][k] << " " ; 
            ss << "]" ; 
        }
    }
    std::string s = ss.str() ; 
    return s ; 
}

template<typename T>
inline std::string Tran<T>::desc() const 
{
    bool only_tlate = false ; 
    std::stringstream ss ; 
    ss << brief(only_tlate, 't' ) << std::endl ; 
    ss << brief(only_tlate, 'v' ) << std::endl ; 
    ss << brief(only_tlate, 'i' ) << std::endl ; 
    std::string s = ss.str() ; 
    return s ; 
}

template<typename T>
bool Tran<T>::checkIsIdentity(char mat, const char* caller, T epsilon)
{
    bool ok = is_identity('i', epsilon); 
    if(!ok)
    {
        T mxdif = maxdiff_from_identity('i'); 
        std::cerr 
            << "stran.h : Tran::checkIsIdentity FAIL : "
            << " caller " << caller 
            << " epsilon " << epsilon
            << " mxdif_from_identity " << mxdif
            << std::endl 
            ;  

        if(getenv("stran_checkIsIdentity_SIGINT")) std::raise(SIGINT); 
    }
    return ok ; 
}

template<typename T>
Tran<T>* Tran<T>::ConvertToTran(const qat4* q_, T epsilon )
{
    qat4 q(q_->cdata()); 
    q.clearIdentity();  

    glm::tmat4x4<T> tran = MatFromQat(&q) ; 
    glm::tmat4x4<T> itra = glm::inverse(tran) ;     
    Tran<T>* tr = new Tran<T>(tran, itra) ; 
    bool ok = tr->checkIsIdentity('i', "ConvertToTran"); 
    if(!ok) std::cerr << "stran.h Tran::ConvertToTran checkIsIdentity FAIL " << std::endl ; 

    return tr ; 
}

template<typename T>
Tran<T>* Tran<T>::ConvertFromQat(const qat4* q_, T epsilon )
{
    qat4 q(q_->cdata()); 
    q.clearIdentity();  

    glm::tmat4x4<T> tran = MatFromQat(&q) ; 
    glm::tmat4x4<T> itra = glm::inverse(tran) ;     
    Tran<T>* tr = new Tran<T>(tran, itra) ; 
    bool ok = tr->checkIsIdentity('i', "ConvertFromQat"); 
    if(!ok) std::cerr << "stran.h Tran::ConvertFromQat checkIsIdentity FAIL " << std::endl ; 
    return tr ; 
}




template<typename T>
Tran<T>* Tran<T>::ConvertFromData(const T* data )
{
    glm::tmat4x4<T> tran = MatFromData(data) ; 
    glm::tmat4x4<T> itra = glm::inverse(tran) ;     
    Tran<T>* tr = new Tran<T>(tran, itra) ; 
    bool ok = tr->checkIsIdentity('i', "ConvertFromData"); 
    if(!ok) std::cerr << "stran.h Tran::ConvertFromData checkIsIdentity FAIL " << std::endl ; 
    return tr ; 
}





template<typename T>
const qat4* Tran<T>::Invert( const qat4* q, T epsilon )
{

    int ins_idx,  gas_idx, sensor_identifier, sensor_index ; 
    q->getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  

    Tran<T>* tr = ConvertToTran(q) ; 

    qat4* v = ConvertFrom(tr->v);
    v->setIdentity(ins_idx, gas_idx, sensor_identifier, sensor_index ) ;

    return v ; 
}


template<typename T>
Tran<T>* Tran<T>::FromPair(const qat4* t, const qat4* v, T epsilon ) // static
{
    glm::tmat4x4<T> tran = MatFromQat(t) ; 
    glm::tmat4x4<T> itra = MatFromQat(v) ; 
    Tran<T>* tr = new Tran<T>(tran, itra) ; 
    bool ok = tr->checkIsIdentity('i', "FromPair", epsilon ); 
    if(!ok) 
    {
         std::cerr << "stran.h Tran::FromPair checkIsIdentity FAIL " << std::endl ; 
         const char* path = "/tmp/stran_FromPair_checkIsIdentity_FAIL.npy" ; 
         std::cerr << "stran.h save to path " << path << std::endl ; 
         tr->save_(path); 
    }
    return tr ; 
}

template<typename T>
glm::tmat4x4<T> Tran<T>::MatFromQat(const qat4* q )  // static
{
    const float* q_data = q->cdata();
    glm::tmat4x4<T> tran(1.);
    T* tran_ptr = glm::value_ptr(tran) ;
    for(int i=0 ; i < 16 ; i++) tran_ptr[i] = T(q_data[i]) ; 
    return tran ; 
}

template<typename T>
glm::tmat4x4<T> Tran<T>::MatFromData(const T* data)  // static
{
    glm::tmat4x4<T> tran(1.);
    T* tran_ptr = glm::value_ptr(tran) ;
    for(int i=0 ; i < 16 ; i++) tran_ptr[i] = data[i] ; 
    return tran ; 
}
















 

/**
Tran::ConvertFrom
-------------------

With T=double will narrow to floats within qat4 

**/

template<typename T>
qat4* Tran<T>::ConvertFrom(const glm::tmat4x4<T>& transform )
{
    const T* ptr = glm::value_ptr(transform) ;

    float ff[16] ; 

    for(int i=0 ; i < 16 ; i++ ) ff[i] = float(ptr[i]) ; 

    return new qat4(ff) ; 
}

template<typename T>
void Tran<T>::write(T* dst, unsigned num_values) const 
{
    unsigned matrix_values = 4*4 ; 
    assert( num_values == 3*matrix_values ); 
  
    unsigned matrix_bytes = matrix_values*sizeof(T) ; 
    char* dst_bytes = (char*)dst ; 

    memcpy( dst_bytes + 0*matrix_bytes , (char*)glm::value_ptr(t), matrix_bytes );
    memcpy( dst_bytes + 1*matrix_bytes , (char*)glm::value_ptr(v), matrix_bytes );
    memcpy( dst_bytes + 2*matrix_bytes , (char*)glm::value_ptr(i), matrix_bytes );
}

template<typename T>
NP* Tran<T>::as_array()const 
{
    NP* a = NP::Make<T>(3, 4, 4 );  
    unsigned num_values = 3*4*4 ; 
    write( a->values<T>(), num_values ) ; 
    return a ; 
}

template<typename T>
void Tran<T>::save(const char* dir, const char* name) const 
{
    NP* a = as_array(); 
    a->save(dir, name);
}
template<typename T>
void Tran<T>::save_(const char* path) const 
{
    NP* a = as_array(); 
    a->save(path);
}






/**
Tran::Apply
-------------

Applies the transform *count* times using 4th component w 
(w is usually 1. for transforming as a position or 0. for transforming as a direction).

**/

template<typename T>
void Tran<T>::Apply( const glm::tmat4x4<T>& tr,  T* p0, T w, unsigned count, unsigned stride, unsigned offset, bool normalize )
{
    for(unsigned i=0 ; i < count ; i++)
    {
        T* a_ = p0 + i*stride + offset ;
 
        glm::tvec4<T> a(a_[0],a_[1],a_[2],w); 
        glm::tvec4<T> ta = tr * a ;
        glm::tvec3<T> ta3(ta); 
        glm::tvec3<T> nta3 = normalize ? glm::normalize(ta3) : ta3  ; 
        T* ta_ = glm::value_ptr( nta3 ) ; 

        for(unsigned j=0 ; j < 3 ; j++) a_[j] = ta_[j] ; 

        //std::cout << " apply: a " << glm::to_string( a ) << std::endl ; 
        //std::cout << " apply: ta= " << glm::to_string( ta ) << std::endl ; 
    }
}

/**
Tran::ApplyToFloat : inplace application of transform 
----------------------------------------------------

Unlike the above Tran::apply this performs the multiplication 
in the T precision even though the input is in float precision. 
The is pragmatic approach, as it will often be preferable to use a 
double precision transform and single precision array.

So it is better to use "Tran::apply_" with a float input 
 
**/


template<typename T>
void Tran<T>::ApplyToFloat( const glm::tmat4x4<T>& tr, float* p0, float w, unsigned count, unsigned stride, unsigned offset, bool normalize ) 
{
    for(unsigned i=0 ; i < count ; i++)
    {
        float* a_ = p0 + i*stride + offset ;
 
        // widening copy pos/mom/pol into tvec4 using w=1./0./0. for pos/mom/pol
        glm::tvec4<T> a(0.,0.,0.,0.)  ;
        T* aa = glm::value_ptr(a) ; 
        for(unsigned j=0 ; j < 3 ; j++) aa[j] = T(a_[j]) ;  // potentially widen 
        aa[3] = T(w) ; 

        // left-multiply the "t" transform with the pos/mom/pol
        // tvec3 normalize when chosen 
        glm::tvec4<T> ta = tr * a ;
        glm::tvec3<T> ta3(ta); 
        glm::tvec3<T> nta3 = normalize ? glm::normalize(ta3) : ta3  ; 
        T* ta_ = glm::value_ptr( nta3 ) ; 

        // copy back into src array 
        for(unsigned j=0 ; j < 3 ; j++) a_[j] = float(ta_[j]) ;   // potentially narrow  

        //std::cout << " apply_: a " << glm::to_string( a ) << std::endl ; 
        //std::cout << " apply_: ta= " << glm::to_string( ta ) << std::endl ; 
    }
}


template<typename T>
void Tran<T>::photon_transform( NP* ph, bool normalize ) const 
{
    T one(1.); 
    T zero(0.); 

    assert( ph->has_shape(-1,4,4) ); 
    unsigned count  = ph->shape[0] ; 
    unsigned stride = 4*4 ; 

    if( ph->ebyte == sizeof(T) )
    {
        T* p0 = ph->values<T>(); 
        Apply( t, p0, one,  count, stride, 0, false );  // transform pos as position
        Apply( t, p0, zero, count, stride, 4, normalize );  // transform mom as direction
        Apply( t, p0, zero, count, stride, 8, normalize );  // transform pol as direction
    }
    else if( ph->ebyte == 4 && sizeof(T) == 8 ) 
    {
        float* p0 = ph->values<float>(); 
        ApplyToFloat( t, p0, one,  count, stride, 0, false );  // transform pos as position
        ApplyToFloat( t, p0, zero, count, stride, 4, normalize );  // transform mom as direction
        ApplyToFloat( t, p0, zero, count, stride, 8, normalize );  // transform pol as direction
    }
}

/**
Tran::PhotonTransform
-----------------------

Test::

   cd ~/opticks/CSG/tests  
   ./CSGFoundry_getFrame_Test.sh

Note that the returned transformed photon array is always in double precision. 

**/

template<typename T>
NP* Tran<T>::PhotonTransform( const NP* ph, bool normalize, const Tran<T>* t ) // static 
{
    NP* b = NP::MakeWideIfNarrow(ph) ; 
    t->photon_transform(b, normalize); 
    assert( b->ebyte == 8 ); 
    return b ; 
}


/**
Tran::AddTransform
---------------------

* The option string determines the meaning of the a,b,c vectors
  and flip lowercase/UPPERCASE flip:true/FALSE




TR/tr
    Tran::Place(a,b,c,flip)
R/r
    Tran::RotateA2B(a,b, flip)   c is ignored 
T/t

D
    

**/


template<typename T>
inline void Tran<T>::AddTransform( T* ttk, const char* opt, const glm::tvec3<T>& a, const glm::tvec3<T>& b, const glm::tvec3<T>& c )
{
    for(unsigned i=0 ; i < 16 ; i++) ttk[i] = T(0.) ; // start zeroed


    if(strcmp(opt,"TR") == 0 || strcmp(opt,"tr") == 0 )
    {
        bool flip = strcmp(opt,"tr") == 0 ; 
        glm::tmat4x4<T> tr = stra<T>::Place( a, b, c, flip );  
        T* src = glm::value_ptr(tr) ; 
        //std::cout << Desc(src, 16) << std::endl ; 
        memcpy( ttk, src, 16*sizeof(T) ); 
    }
    else if(strcmp(opt,"R") == 0 || strcmp(opt,"r") == 0)
    {
        bool flip = strcmp(opt,"r") == 0 ; 
        glm::tmat4x4<T> tr = stra<T>::RotateA2B( a, b, flip );  
        T* src = glm::value_ptr(tr) ; 
        memcpy( ttk, src, 16*sizeof(T) ); 
    }
    else if(strcmp(opt,"T") == 0 || strcmp(opt,"t") == 0)
    {
        bool flip = strcmp(opt,"t") == 0 ; 
        glm::tmat4x4<T> tr = stra<T>::Translate(c, flip );  
        T* src = glm::value_ptr(tr) ; 
        memcpy( ttk, src, 16*sizeof(T) ); 
    }
    else if(strcmp(opt,"D")== 0)
    {
        for(int l=0 ; l < 3 ; l++) ttk[4*0+l] = a[l] ; 
        for(int l=0 ; l < 3 ; l++) ttk[4*1+l] = b[l] ; 
        for(int l=0 ; l < 3 ; l++) ttk[4*2+l] = c[l] ; 
    }
    else
    {
        std::cout << "Tran::AddTransform : ERROR opt is not handled [" << opt << "]" << std::endl ; 
    }
}




template struct Tran<float> ;
template struct Tran<double> ;



