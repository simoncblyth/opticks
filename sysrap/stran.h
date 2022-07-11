#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>


/**
Tran
=====

Transform handler that creates inverse transforms at every stage  
loosely based on NPY nmat4triple_

Aim to avoid having to take the inverse eg with glm::inverse 
which inevitably risks numerical issues.

**/

#include "sqat4.h"
#include "NP.hh"
#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"

#include <vector>

template<typename T>
struct Tran
{
    // TODO: on stack ctors 

    static constexpr const T EPSILON = 1e-6 ; 

    static const Tran<T>* make_translate( const T tx, const T ty, const T tz, const T sc);
    static const Tran<T>* make_translate( const T tx, const T ty, const T tz);
    static const Tran<T>* make_identity();
    static const Tran<T>* make_scale(     const T sx, const T sy, const T sz);
    static const Tran<T>* make_rotate(    const T ax, const T ay, const T az, const T angle_deg);

    static const Tran<T>* product(const Tran<T>* a, const Tran<T>* b, bool reverse);
    static const Tran<T>* product(const Tran<T>* a, const Tran<T>* b, const Tran<T>* c, bool reverse);
    static const Tran<T>* product(const std::vector<const Tran<T>*>& tt, bool reverse );

    static Tran<T>* ConvertToTran( const qat4* q, T epsilon=EPSILON ); 
    static Tran<T>* ConvertFromData(const T* data); 

    static const qat4* Invert( const qat4* q, T epsilon=EPSILON ); 
    static Tran<T>* FromPair( const qat4* t, const qat4* v, T epsilon=EPSILON ); // WIDENS from float  
    static glm::tmat4x4<T> MatFromQat( const qat4* q );
    static glm::tmat4x4<T> MatFromData(const T* data );

    static qat4*    ConvertFrom(const glm::tmat4x4<T>& tr ); 

    Tran( const T* transform, const T* inverse ) ;
    Tran( const glm::tmat4x4<T>& transform, const glm::tmat4x4<T>& inverse ) ;

    T    maxdiff_from_identity(char mat='t') const ; 
    bool is_identity(char mat='t', T epsilon=1e-6) const ; 
    std::string brief(bool only_tlate=false, char mat='t', unsigned wid=6, unsigned prec=1) const ;  
    std::string desc() const ; 
    bool checkIsIdentity(char mat='i', const char* caller="caller", T epsilon=EPSILON); 
 
    void write(T* dst, unsigned num_values=3*4*4) const ; 
    void save(const char* dir, const char* name="stran.npy") const ; 

    void apply( T* p0, T w, unsigned count, unsigned stride, unsigned offset ) const ; 
    void apply_( float* p0, float w, unsigned count, unsigned stride, unsigned offset ) const ; 
    void apply( NP* ph ) const ; 
    static NP* Apply( const NP* ph, const Tran<T>* tr ); 


    const T* tdata() const ; 
    const T* vdata() const ; 
 
    glm::tmat4x4<T> t ;  // transform 
    glm::tmat4x4<T> v ;  // inverse  
    glm::tmat4x4<T> i ;  // identity
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

#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>







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

Tran houses paired transforms with their inverse transforms, the product 
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
        const Tran<T>* ii = tt[reverse ? j : i] ;  // with reverse: start from the last (ie root node)
        const Tran<T>* jj = tt[reverse ? i : j] ;  // with reverse: start from the first (ie leaf node)

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
            << "Tran::checkIsIdentity fail from " << caller 
            << " epsilon " << epsilon
            << " mxdif_from_identity " << mxdif
            << std::endl 
            ;  
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
    tr->checkIsIdentity('i', "ConvertToTran"); 
    return tr ; 
}

template<typename T>
Tran<T>* Tran<T>::ConvertFromData(const T* data )
{
    glm::tmat4x4<T> tran = MatFromData(data) ; 
    glm::tmat4x4<T> itra = glm::inverse(tran) ;     
    Tran<T>* tr = new Tran<T>(tran, itra) ; 
    tr->checkIsIdentity('i', "ConvertToTran"); 
    return tr ; 
}





template<typename T>
const qat4* Tran<T>::Invert( const qat4* q, T epsilon )
{
    unsigned ins_idx, gas_idx, ias_idx ;
    q->getIdentity(ins_idx, gas_idx, ias_idx )  ;

    Tran<T>* tr = ConvertToTran(q) ; 

    qat4* v = ConvertFrom(tr->v);
    v->setIdentity(ins_idx, gas_idx, ias_idx ) ;

    return v ; 
}


template<typename T>
Tran<T>* Tran<T>::FromPair(const qat4* t, const qat4* v, T epsilon ) // static
{
    glm::tmat4x4<T> tran = MatFromQat(t) ; 
    glm::tmat4x4<T> itra = MatFromQat(v) ; 
    Tran<T>* tr = new Tran<T>(tran, itra) ; 
    tr->checkIsIdentity('i', "FromPair", epsilon ); 
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
void Tran<T>::save(const char* dir, const char* name) const 
{
    NP* a = NP::Make<T>(3, 4, 4 );  
    unsigned num_values = 3*4*4 ; 
    write( a->values<T>(), num_values ) ; 
    a->save(dir, name);
}

/**
Tran::apply
-------------

Applies the transform *count* times using 4th component w 
(w is usually 1. for transforming as a position or 0. for transforming as a direction).

**/

template<typename T>
void Tran<T>::apply( T* p0, T w, unsigned count, unsigned stride, unsigned offset ) const 
{
    for(unsigned i=0 ; i < count ; i++)
    {
        T* a_ = p0 + i*stride + offset ;
 
        glm::tvec4<T> a(a_[0],a_[1],a_[2],w); 
        glm::tvec4<T> ta = t * a ;
        T* ta_ = glm::value_ptr( ta ) ; 

        for(unsigned j=0 ; j < 3 ; j++) a_[j] = ta_[j] ; 

        //std::cout << " apply: a " << glm::to_string( a ) << std::endl ; 
        //std::cout << " apply: ta= " << glm::to_string( ta ) << std::endl ; 

    }
}

/**
HMM: the above  assumes the float/double of the array is same as the transform
but it will often be preferable to use a double precision transform 
and single precision array 
**/


template<typename T>
void Tran<T>::apply_( float* p0, float w, unsigned count, unsigned stride, unsigned offset ) const 
{
    for(unsigned i=0 ; i < count ; i++)
    {
        float* a_ = p0 + i*stride + offset ;
 
        glm::tvec4<T> a(0.,0.,0.,0.)  ;
        T* aa = glm::value_ptr(a) ; 
        for(unsigned j=0 ; j < 3 ; j++) aa[j] = T(a_[j]) ;  // potentially widen 
        aa[3] = T(w) ; 


        glm::tvec4<T> ta = t * a ;
        T* ta_ = glm::value_ptr( ta ) ; 
        for(unsigned j=0 ; j < 3 ; j++) a_[j] = float(ta_[j]) ;   // potentially narrow  

        //std::cout << " apply_: a " << glm::to_string( a ) << std::endl ; 
        //std::cout << " apply_: ta= " << glm::to_string( ta ) << std::endl ; 
    }
}
 


template<typename T>
void Tran<T>::apply( NP* ph ) const 
{
    T one(1.); 
    T zero(0.); 

    assert( ph->has_shape(-1,4,4) ); 
    unsigned count  = ph->shape[0] ; 
    unsigned stride = 4*4 ; 

    if( ph->ebyte == sizeof(T) )
    {
        T* p0 = ph->values<T>(); 
        apply( p0, one,  count, stride, 0 );  // transform pos as position
        apply( p0, zero, count, stride, 4 );  // transform mom as direction
        apply( p0, zero, count, stride, 8 );  // transform pol as direction
    }
    else if( ph->ebyte == 4 && sizeof(T) == 8 ) 
    {
        float* p0 = ph->values<float>(); 
        apply_( p0, one,  count, stride, 0 );  // transform pos as position
        apply_( p0, zero, count, stride, 4 );  // transform mom as direction
        apply_( p0, zero, count, stride, 8 );  // transform pol as direction
    }


}


template<typename T>
NP* Tran<T>::Apply( const NP* ph, const Tran<T>* t ) // static 
{
    NP* b = ph->copy(); 
    t->apply(b); 
    return b ; 
}





template struct Tran<float> ;
template struct Tran<double> ;



