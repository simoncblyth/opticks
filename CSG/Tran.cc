
#include <sstream>
#include <iomanip>

#include "Tran.h"
#include <glm/gtx/transform.hpp>

#include <glm/gtc/type_ptr.hpp>




template<typename T>
const Tran<T>* Tran<T>::make_translate( const T tx, const T ty, const T tz, const T sc)
{
    glm::tvec3<T> tlate(tx*sc,ty*sc,tz*sc); 
    glm::tmat4x4<T> t = glm::translate(glm::tmat4x4<T>(1.),   tlate ) ;
    glm::tmat4x4<T> v = glm::translate(glm::tmat4x4<T>(1.),  -tlate ) ;
    return new Tran<T>(t, v);    
}

template<typename T>
const Tran<T>* Tran<T>::make_translate( const T tx, const T ty, const T tz)
{
    glm::tvec3<T> tlate(tx,ty,tz); 
    glm::tmat4x4<T> t = glm::translate(glm::tmat4x4<T>(1.),   tlate ) ;
    glm::tmat4x4<T> v = glm::translate(glm::tmat4x4<T>(1.),  -tlate ) ;
    return new Tran<T>(t, v);    
}

template<typename T>
const Tran<T>* Tran<T>::make_scale( const T sx, const T sy, const T sz)
{
    glm::tvec3<T> scal(sx,sy,sz); 
    glm::tvec3<T> isca(1./sx,1./sy,1./sz); 
    glm::tmat4x4<T> t = glm::scale(glm::tmat4x4<T>(1.),   scal ) ;
    glm::tmat4x4<T> v = glm::scale(glm::tmat4x4<T>(1.),   isca ) ;
    return new Tran<T>(t, v);    
}

template<typename T>
const Tran<T>* Tran<T>::make_rotate( const T ax, const T ay, const T az, const T angle_deg)
{
    T angle_rad = glm::pi<T>()*angle_deg/T(180.) ;
    glm::tvec3<T> axis(ax,ay,az); 
    glm::tmat4x4<T> t = glm::rotate(glm::tmat4x4<T>(1.),  angle_rad, axis ) ;
    glm::tmat4x4<T> v = glm::rotate(glm::tmat4x4<T>(1.), -angle_rad, axis ) ;
    return new Tran<T>(t, v);    
}

template<typename T>
const Tran<T>* Tran<T>::product(const Tran<T>* a, const Tran<T>* b, bool reverse)
{
    std::vector<const Tran<T>*> tt ; 
    tt.push_back(a);
    tt.push_back(b);
    return Tran<T>::product( tt, reverse );
}

template<typename T>
const Tran<T>* Tran<T>::product(const Tran<T>* a, const Tran<T>* b, const Tran<T>* c, bool reverse)
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

Use *reverse=true* when the transforms are in reverse heirarchical order, ie when
they have been collected by starting from the leaf node and then following parent 
links back up to the root node. 

**/
template<typename T>
const Tran<T>* Tran<T>::product(const std::vector<const Tran<T>*>& tt, bool reverse )
{
    unsigned ntt = tt.size();
    if(ntt==0) return NULL ; 
    if(ntt==1) return tt[0] ;

    glm::tmat4x4<T> t(T(1)) ;
    glm::tmat4x4<T> v(T(1)) ;

    for(unsigned i=0,j=ntt-1 ; i < ntt ; i++,j-- )
    {
        // inclusive indices:
        //     i: 0 -> ntt - 1      ascending 
        //     j: ntt - 1 -> 0      descending (from last transform down to first)
        //
        const Tran<T>* ii = tt[reverse ? j : i] ;  // with reverse: start from the last (ie root node)
        const Tran<T>* jj = tt[reverse ? i : j] ;  // with reverse: start from the first (ie leaf node)

        t *= ii->t ;  
        v *= jj->v ;    // inverse-transform product in opposite order
    }
    return new Tran<T>(t, v) ;
}





template<typename T>
Tran<T>::Tran( const T* transform, const T* inverse ) 
    :   
    t(glm::make_mat4x4<T>(transform)), 
    v(glm::make_mat4x4<T>(inverse)),
    i(t*v)
{
} 

template<typename T>
Tran<T>::Tran( const glm::tmat4x4<T>& transform, const glm::tmat4x4<T>& inverse ) 
    :   
    t(transform), 
    v(inverse),
    i(transform*inverse)
{
} 


template<typename T>
const T*  Tran<T>::tdata() const 
{
    return glm::value_ptr(t) ; 
}

template<typename T>
const T*  Tran<T>::vdata() const 
{
    return glm::value_ptr(v) ; 
}

 


template<typename T>
bool Tran<T>::is_identity(char mat, T epsilon) const 
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
std::string Tran<T>::brief(bool only_tlate, char mat, unsigned wid, unsigned prec) const 
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




template struct Tran<float> ;
template struct Tran<double> ;

