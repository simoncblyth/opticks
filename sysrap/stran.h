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
#include "glm/glm.hpp"

#include <vector>

template<typename T>
struct Tran
{
    // TODO: on stack ctors 

    static const Tran<T>* make_translate( const T tx, const T ty, const T tz, const T sc);
    static const Tran<T>* make_translate( const T tx, const T ty, const T tz);
    static const Tran<T>* make_identity();
    static const Tran<T>* make_scale(     const T sx, const T sy, const T sz);
    static const Tran<T>* make_rotate(    const T ax, const T ay, const T az, const T angle_deg);

    static const Tran<T>* product(const Tran<T>* a, const Tran<T>* b, bool reverse);
    static const Tran<T>* product(const Tran<T>* a, const Tran<T>* b, const Tran<T>* c, bool reverse);
    static const Tran<T>* product(const std::vector<const Tran<T>*>& tt, bool reverse );

    static Tran<T>* ConvertToTran( const qat4* q ); 
    static Tran<T>* FromPair( const qat4* t, const qat4* v, T epsilon=1e-6 ); 
    static glm::tmat4x4<T> MatFromQat(const qat4* q );
    static qat4*    ConvertFrom(const glm::tmat4x4<T>& tr ); 

    Tran( const T* transform, const T* inverse ) ;
    Tran( const glm::tmat4x4<T>& transform, const glm::tmat4x4<T>& inverse ) ;

    bool is_identity(char mat='t', T epsilon=1e-6) const ; 
    std::string brief(bool only_tlate=false, char mat='t', unsigned wid=6, unsigned prec=1) const ;  
 
    void write(T* dst, unsigned num_values=3*4*4) const ; 

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
inline bool Tran<T>::is_identity(char mat, T epsilon) const 
{
    const glm::mat4& m = mat == 't' ? t : ( mat == 'v' ? v : i ) ; 
    unsigned mismatch = 0 ; 
    for(int j=0 ; j < 4 ; j++ ) 
    for(int k=0 ; k < 4 ; k++ ) 
    {
        T val = m[j][k] ; 
        T xval = j == k ? T(1) : T(0) ; 
        if( std::abs( val - xval ) > epsilon )  mismatch += 1 ; 
    }
    return mismatch == 0 ; 
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
        const glm::mat4& m = mat == 't' ? t : ( mat == 'v' ? v : i ) ; 
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
Tran<T>* Tran<T>::ConvertToTran(const qat4* q )
{
    const float* qdata = q->cdata(); 

    glm::tmat4x4<T> tran(1.);

    T* ptr = glm::value_ptr(tran) ;

    for(int i=0 ; i < 16 ; i++) ptr[i] = T(qdata[i]) ; 

    glm::tmat4x4<T> itra = glm::inverse(tran) ;     

    return new Tran<T>(tran, itra) ; 
}


template<typename T>
Tran<T>* Tran<T>::FromPair(const qat4* t, const qat4* v, T epsilon ) // static
{
    glm::tmat4x4<T> tran = MatFromQat(t) ; 
    glm::tmat4x4<T> itra = MatFromQat(v) ; 
    Tran<T>* tr = new Tran<T>(tran, itra) ; 

    bool ok = tr->is_identity('i', epsilon); 

    if(!ok)
    {
        std::cerr 
            << " Tran::FromPair is_identity fail with epsilon " << epsilon
            << std::endl 
            ;  
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
 


template struct Tran<float> ;
template struct Tran<double> ;



