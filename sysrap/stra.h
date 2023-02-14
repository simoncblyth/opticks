#pragma once
/**
stra.h
========

Following some investigations of Geant4 transform handling 
and noting that inverses are being done their at source
have concluded that dealing with transforms together with 
their inverses is not worth the overhead and complication. 
Of course inverting should be minimized. 

Hence are bringing over functionality from stran.h as its needed 
in new code. 

**/

#include <array>

#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>



template<typename T>
struct stra
{
    static constexpr const bool VERBOSE = false ;  

    static std::string Desc(const glm::tmat4x4<T>& t, const glm::tmat4x4<T>& v, const char* tl, const char* vl ); 
    static std::string Desc(
        const glm::tmat4x4<double>& a, 
        const glm::tmat4x4<double>& b, 
        const glm::tmat4x4<double>& c, 
        const char* al, 
        const char* bl, 
        const char* cl); 

    static std::string Desc(const T* aa, const T* bb, const T* cc, const char* al, const char* bl, const char* cl) ; 

    static std::string Desc(const glm::tmat4x4<T>& tr);  // WHY WERE THESE BY VALUE ?
    static std::string Desc(const glm::tvec4<T>& t); 
    static std::string Desc(const glm::tvec3<T>& t); 
    static std::string Desc(const T* tt, int num); 

    static glm::tmat4x4<T> FromData(const T* data );

    static glm::tmat4x4<T> Translate( const T tx, const T ty, const T tz, const T sc, bool flip=false ); 
    static glm::tmat4x4<T> Translate( const glm::tvec3<T>& tlate, bool flip=false ); 

    static glm::tmat4x4<T> Rotate( const T ax, const T ay, const T az, const T angle_deg, bool flip=false) ; 

    static glm::tvec3<T>   LeastParallelAxis(const glm::tvec3<T>& a ); 
    static glm::tmat4x4<T> RotateA2B_nonparallel( const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip ); 
    static glm::tmat4x4<T> RotateA2B_parallel(    const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip ); 
    static glm::tmat4x4<T> RotateA2B(             const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip ); 

    static glm::tmat4x4<T> Place(                 const glm::tvec3<T>& a, const glm::tvec3<T>& b, const glm::tvec3<T>& c, bool flip); 
}; 


template<typename T>
std::string stra<T>::Desc(const glm::tmat4x4<T>& t, const glm::tmat4x4<T>& v, const char* tl, const char* vl )  // static
{
    glm::tmat4x4<double> tv = t*v ; 

    std::stringstream ss ; 
    ss << tl << "*" << vl ; 
    std::string tv_l = ss.str(); 

    return Desc( t, v, tv,  tl, vl, tv_l.c_str() ); 
}

template<typename T>
std::string stra<T>::Desc(
    const glm::tmat4x4<double>& a, 
    const glm::tmat4x4<double>& b, 
    const glm::tmat4x4<double>& c, 
    const char* al, 
    const char* bl, 
    const char* cl)
{
    return Desc(glm::value_ptr(a),glm::value_ptr(b),glm::value_ptr(c),al,bl,cl) ; 
}

template<typename T>
std::string stra<T>::Desc(const T* aa, const T* bb, const T* cc, const char* al, const char* bl, const char* cl)
{
    std::stringstream ss ; 
    ss << "\n" ; 

    if(al) ss << " " << std::setw(54) << std::left << al ; 
    if(bl) ss << " " << std::setw(54) << std::left << bl ; 
    if(cl) ss << " " << std::setw(54) << std::left << cl ; 

    for(int i=0 ; i < 4 ; i++) 
    {
        ss << "\n" ; 
        if(aa) for(int j=0 ; j < 4 ; j++) ss << " " << std::fixed << std::setw(10) << std::setprecision(4) << aa[i*4+j] ; 
        ss << " " << std::setw(10) << " " ; 
        if(bb) for(int j=0 ; j < 4 ; j++) ss << " " << std::fixed << std::setw(10) << std::setprecision(4) << bb[i*4+j] ; 
        ss << " " << std::setw(10) << " " ; 
        if(cc) for(int j=0 ; j < 4 ; j++) ss << " " << std::fixed << std::setw(10) << std::setprecision(4) << cc[i*4+j] ; 
    }
    ss << "\n" ; 
    std::string s = ss.str(); 
    return s ; 
}


template<typename T>
std::string stra<T>::Desc(const glm::tmat4x4<T>& tr)
{
    const T* tt = glm::value_ptr(tr); 
    return Desc(tt, 16 ); 
}
template<typename T>
std::string stra<T>::Desc(const glm::tvec4<T>& t)
{
    const T* tt = glm::value_ptr(t); 
    return Desc(tt, 4 ); 
}
template<typename T>
std::string stra<T>::Desc(const glm::tvec3<T>& t)
{
    const T* tt = glm::value_ptr(t); 
    return Desc(tt, 3 ); 
}
template<typename T>
std::string stra<T>::Desc(const T* tt, int num)
{
    std::stringstream ss ; 
    for(int i=0 ; i < num ; i++) 
        ss 
            << ( i % 4 == 0 && num > 4 ? ".\n" : "" ) 
            << " " << std::fixed << std::setw(10) << std::setprecision(4) << tt[i] 
            << ( i == num-1 && num > 4 ? ".\n" : "" ) 
            ;

    std::string s = ss.str(); 
    return s ; 
}



template<typename T>
glm::tmat4x4<T> stra<T>::FromData(const T* data)  // static
{
    glm::tmat4x4<T> tran(1.);
    memcpy( glm::value_ptr(tran), data, 16*sizeof(T) ); 
    return tran ; 
}





template<typename T>
inline glm::tmat4x4<T> stra<T>::Translate(  const T tx, const T ty, const T tz, const T sc, bool flip )
{
    glm::tvec3<T> tlate(tx*sc,ty*sc,tz*sc); 
    return Translate(tlate, flip) ; 
}

template<typename T>
inline glm::tmat4x4<T> stra<T>::Translate( const glm::tvec3<T>& tlate, bool flip )
{
    glm::tmat4x4<T> tr = glm::translate(glm::tmat4x4<T>(1.), tlate ) ;
    return flip ? glm::transpose(tr) : tr ; 
}

template<typename T>
inline glm::tmat4x4<T> stra<T>::Rotate( const T ax, const T ay, const T az, const T angle_deg, bool flip)
{
    T angle_rad = glm::pi<T>()*angle_deg/T(180.) ;
    glm::tvec3<T> axis(ax,ay,az); 
    glm::tmat4x4<T> tr = glm::rotate(glm::tmat4x4<T>(1.), ( flip ? -1. : 1. )* angle_rad, axis ) ;
    return tr ; 
}





template<typename T>
inline glm::tvec3<T> stra<T>::LeastParallelAxis( const glm::tvec3<T>& a )
{
    glm::tvec3<T> aa( glm::abs(a) ); 
    glm::tvec3<T> lpa(0.); 

    if( aa.x <= aa.y && aa.x <= aa.z )
    {    
        lpa.x = 1.f ; 
    }    
    else if( aa.y <= aa.x && aa.y <= aa.z )
    {    
        lpa.y = 1.f ; 
    }    
    else 
    {    
        lpa.z = 1.f ; 
    }    

    if(VERBOSE) std::cout 
        << "stra::LeastParallelAxis"
        << " a " <<  glm::to_string(a)
        << " aa " <<  glm::to_string(aa)
        << " lpa " << glm::to_string(lpa)
        << std::endl 
        ;

    return lpa ; 
}



/**
stra::RotateA2B
----------------------

See ana/make_rotation_matrix.py 

* http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf
  "Efficiently Building a Matrix to Rotate One Vector To Another"
  Tomas Moller and John F Hughes 

* ~/opticks_refs/Build_Rotation_Matrix_vec2vec_Moller-1999-EBA.pdf

Found this paper via thread: 

* https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

Another very long discussion of rotation matrices by Rebecca Brannon:

* https://my.mech.utah.edu/~brannon/public/rotation.pdf
* ~/opticks_refs/utah_brannon_opus_rotation.pdf

**/

template<typename T>
inline glm::tmat4x4<T> stra<T>::RotateA2B_nonparallel(const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip )
{
    T one(1.); 
    T zero(0.); 

    T c = glm::dot(a,b); 
    T h = (one - c)/(one - c*c); 

    glm::tvec3<T> v = glm::cross(a, b) ;  
    T vx = v.x ; 
    T vy = v.y ; 
    T vz = v.z ; 

    if(VERBOSE) std::cout 
        << "stra::RotateA2B_nonparallel"
        << " a " << glm::to_string(a)
        << " b " << glm::to_string(b)
        << " c " << std::fixed << std::setw(10) << std::setprecision(5) << c 
        << " h " << std::fixed << std::setw(10) << std::setprecision(5) << h
        << " vx " << std::fixed << std::setw(10) << std::setprecision(5) << vx
        << " vy " << std::fixed << std::setw(10) << std::setprecision(5) << vy
        << " vz " << std::fixed << std::setw(10) << std::setprecision(5) << vz
        << std::endl 
        ;


    T hxx = h*vx*vx ; 
    T hyy = h*vy*vy ; 
    T hzz = h*vz*vz ; 

    T hxy = h*vx*vy ; 
    T hxz = h*vx*vz ; 
    T hyz = h*vy*vz ; 

    T f = flip ? -1. : 1. ;   // flip:true is equivalent to transposing the matrix

    std::array<T, 16> vals = {{   
          c + hxx    , hxy - f*vz   ,  hxz + f*vy  , zero, 
          hxy + f*vz , c + hyy      ,  hyz - f*vx  , zero,
          hxz - f*vy , hyz + f*vx   ,  c + hzz     , zero,
          zero       , zero         ,  zero        , one         
    }}; 

    return glm::make_mat4x4<T>( vals.data() ); 
}


template<typename T>
inline glm::tmat4x4<T> stra<T>::RotateA2B_parallel(const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip )
{
    glm::tvec3<T> x = LeastParallelAxis(a); 
    glm::tvec3<T> u = x - a ; 
    glm::tvec3<T> v = x - b  ;    

    if(VERBOSE) std::cout 
        << "Trab::RotateA2B_parallel" 
        << std::endl 
        << " x         " << glm::to_string(x) << std::endl  
        << " u = x - a " << glm::to_string(u) << std::endl  
        << " v = x - b " << glm::to_string(v) << std::endl  
        ;


    T uu = glm::dot(u, u); 
    T vv = glm::dot(v, v); 
    T uv = glm::dot(u, v); 

    std::array<T, 16> vals ; 
    vals.fill(0.) ; 

    for(int i=0 ; i < 3 ; i++) for(int j=0 ; j < 3 ; j++) 
    {
        int idx = flip == false ? i*4+j : j*4 + i ;  
        vals[idx] = ( i == j ? 1. : 0.)  - 2.*u[i]*u[j]/uu -2.*v[i]*v[j]/vv + 4.*uv*v[i]*u[j]/(uu*vv) ; 

    }
    vals[15] = 1. ; 

    return glm::make_mat4x4<T>( vals.data() ); 
}


/**
stra::RotateA2B
------------------

Form a rotation transform that rotates vector a to vector b with proper handling 
of parallel or anti-parallel edge cases when the absolute dot product of a and b 
is close to 1.  

**/

template<typename T>
inline glm::tmat4x4<T> stra<T>::RotateA2B(const glm::tvec3<T>& a, const glm::tvec3<T>& b, bool flip)
{
    T c = glm::dot(a,b); 
    return std::abs(c) < 0.99 ? RotateA2B_nonparallel(a,b,flip) : RotateA2B_parallel(a,b,flip) ; 
}



/**
stra::Place
-------------

Create a transform combining rotation from a to b and translation of c 
The flip argument controls transposes being applied to the rotation 
matrix and the order of multiplication of the rotation and the translation. 

HMM: not sure regards the "tla*rot" vs "rot*tla"  order flip : 
it depends on whether will using the transform to right or left multiply.
 
BUT what is certain is that the placement transform needs to rotate first and then translate 
The complexity comes from different packages needing different layouts, 
so generally have to experiment to get things to work. 
**/

template<typename T>
inline glm::tmat4x4<T> stra<T>::Place(const glm::tvec3<T>& a, const glm::tvec3<T>& b, const glm::tvec3<T>& c, bool flip )
{
    glm::tmat4x4<T> rot = RotateA2B(a,b,flip) ; 
    glm::tmat4x4<T> tla = Translate(c) ;              // this has the translation in last row
    glm::tmat4x4<T> tra = flip == true ? tla * rot : rot * tla ;   

    return tra ; 
}





