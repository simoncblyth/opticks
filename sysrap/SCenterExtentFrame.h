#pragma once
/**
SCenterExtentFrame
===================

See also 

* ana/tangential.py
* ana/pvprim.py
* ana/pvprim1.py
* ana/symtran.py
* ana/tangential.cc

**/
#include <glm/glm.hpp>



#include "SYSRAP_API_EXPORT.hh"

template<typename T>
struct SYSRAP_API SCenterExtentFrame 
{
    // convert between coordinate systems
    static void CartesianToSpherical( glm::tvec3<T>& radius_theta_phi, const glm::tvec4<T>& xyzw ); 
    static void SphericalToCartesian( glm::tvec4<T>& xyzw, const glm::tvec3<T>& radius_theta_phi );

    // rotation matrices between conventional XYZ and tangential RTP cartesian frames 
    static glm::tmat4x4<T> XYZ_to_RTP( T theta, T phi );
    static glm::tmat4x4<T> RTP_to_XYZ( T theta, T phi );

    SCenterExtentFrame( float  _cx, float  _cy, float  _cz, float  _extent, bool rtp_tangential ) ; 
    SCenterExtentFrame( double _cx, double _cy, double _cz, double _extent, bool rtp_tangential ) ; 

    void init();  
    void dump(const char* msg="SCenterExtentFrame::dump") const ; 

    glm::tvec4<T> ce ;    // center extent 
    bool          rtp_tangential ; 

    glm::tvec3<T> rtp ;   // radius_theta_phi 
    glm::tvec4<T> xyzw ;  

    glm::tmat4x4<T> scale ; 
    glm::tmat4x4<T> iscale ; 
    glm::tmat4x4<T> translate ; 
    glm::tmat4x4<T> itranslate ; 
    glm::tmat4x4<T> rotate ; 
    glm::tmat4x4<T> irotate ; 

    glm::tmat4x4<T> model2world ; 
    glm::tmat4x4<T> world2model ; 

    const T* model2world_data ; 
    const T* world2model_data ; 


}; 

#include <iostream>
#include <array>
#include "SPresent.h"


/**
SCenterExtentFrame::CartesianToSpherical
-----------------------------------------

Obtains (radius, theta, phi) from the xyz of the xyzw (NB w is not used). 

**/

template<typename T>
inline void SCenterExtentFrame<T>::CartesianToSpherical( glm::tvec3<T>& radius_theta_phi, const glm::tvec4<T>& xyzw ) // static
{
    const T x = xyzw.x ; 
    const T y = xyzw.y ; 
    const T z = xyzw.z ; 
    const T radius =  sqrt( x*x + y*y + z*z ) ; 
    const T zero(0.) ; 

    const T theta = radius == zero ? zero : acos( z/radius );  
    const T phi = atan2(y, x) ; 

    radius_theta_phi.x = radius ; 
    radius_theta_phi.y = theta ; 
    radius_theta_phi.z = phi ; 
}
/**
SCenterExtentFrame::SphericalToCartesian
-------------------------------------------

Obtains xyz from the spherical coordinates. NB the xyzw.w is set to 1.

**/

template<typename T>
inline void SCenterExtentFrame<T>::SphericalToCartesian( glm::tvec4<T>& xyzw, const glm::tvec3<T>& radius_theta_phi ) // static
{
    const T radius = radius_theta_phi.x ;
    const T theta  = radius_theta_phi.y ;
    const T phi    = radius_theta_phi.z ;
    const T one(1.) ; 

    const T x = radius*sin(theta)*cos(phi) ; 
    const T y = radius*sin(theta)*sin(phi) ; 
    const T z = radius*cos(theta) ; 
    const T w = one ; 

    xyzw.x = x ; 
    xyzw.y = y ; 
    xyzw.z = z ; 
    xyzw.w = w ; 
}

/**
SCenterExtentFrame::XYZ_to_RTP
--------------------------------

Returns rotation matrix needed for transforming cartesian (x,y,z) 
to a specific tangential frame at a (theta, phi) point on an imaginary sphere. 
NB just the rotation here, no translation 

**/

template<typename T>
inline glm::tmat4x4<T> SCenterExtentFrame<T>::XYZ_to_RTP( T theta, T phi )  // static
{
    std::array<T, 16> _rot = {{
        T(sin(theta)*cos(phi)),  T(sin(theta)*sin(phi)),  T(cos(theta)),  T(0.) ,
        T(cos(theta)*cos(phi)),  T(cos(theta)*sin(phi)),  T(-sin(theta)), T(0.) ,
        T(-sin(phi)),            T(cos(phi)),             T(0.),          T(0.) ,
        T( 0.),                  T(0.),                   T(0.),          T(1.)
      }} ;
    return glm::make_mat4x4<T>(_rot.data()) ;
}


template<typename T>
inline glm::tmat4x4<T> SCenterExtentFrame<T>::RTP_to_XYZ(  T theta, T phi ) // static
{
    std::array<T, 16> _iro = {{
        T(sin(theta)*cos(phi)),   T(cos(theta)*cos(phi)),  T(-sin(phi)),   T(0.) ,
        T(sin(theta)*sin(phi)),   T(cos(theta)*sin(phi)),  T(cos(phi)),    T(0.) ,
        T(cos(theta)),            T(-sin(theta)),          T(0.),          T(0.) ,
        T(0.),                    T(0.),                   T(0.),          T(1.)
      }} ;
    return glm::make_mat4x4<T>(_iro.data()) ;
}

template<typename T>
inline SCenterExtentFrame<T>::SCenterExtentFrame( float _cx, float _cy, float _cz, float _extent, bool _rtp_tangential  )
    :
    ce(_cx, _cy, _cz, _extent),
    rtp_tangential(_rtp_tangential)
{
    init(); 
}

template<typename T>
inline SCenterExtentFrame<T>::SCenterExtentFrame( double _cx, double _cy, double _cz, double _extent, bool _rtp_tangential  )
    :
    ce(_cx, _cy, _cz, _extent),
    rtp_tangential(_rtp_tangential)
{
    init(); 
}

template<typename T>
inline void SCenterExtentFrame<T>::init()
{ 
    CartesianToSpherical(rtp, ce);     // ce.w the extent is not used here, just the center ce.xyz
    SphericalToCartesian(xyzw, rtp); 

    const T extent = ce.w ; 

    glm::tvec3<T> sc(extent) ;  
    glm::tvec3<T> isc(T(1.)/extent) ;  
    glm::tvec3<T> tr(ce) ;  

    T theta = rtp.y ; 
    T phi = rtp.z ; 
    rotate     = rtp_tangential ? XYZ_to_RTP(theta, phi) : glm::tmat4x4<T>(1.0) ; 
    irotate    = rtp_tangential ? RTP_to_XYZ(theta, phi) : glm::tmat4x4<T>(1.0) ; 
    scale      = glm::scale(     glm::tmat4x4<T>(1.), sc ) ; 
    iscale     = glm::scale(     glm::tmat4x4<T>(1.), isc ) ; 
    translate  = glm::translate( glm::tmat4x4<T>(1.), tr ) ; 
    itranslate = glm::translate( glm::tmat4x4<T>(1.), -tr ) ; 

    world2model = irotate * iscale * itranslate ; 
    model2world = translate * scale * rotate ; 

    world2model_data = glm::value_ptr(world2model) ; 
    model2world_data = glm::value_ptr(model2world) ; 

    /**
    From Composition::setCenterExtent 

    1409     m_world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);    
    1410     m_model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);

    CANNOT GET THAT WAY OF DOING THINGS TO WORK WITH ROTATION : SO ADOPTED MORE EXPLICIT MULTIPLICATION OF MATRIX FORM
    **/
}

template<typename T>
inline void SCenterExtentFrame<T>::dump(const char* msg) const 
{
    std::cout << msg << std::endl ; 
    std::cout << SPresent( ce  , "ce") << " " << ( rtp_tangential ? "RTP_TANGENTIAL" : "ORDINARY" ) << std::endl ; 
    std::cout << SPresent( rtp , "rtp") << std::endl ;  
    std::cout << SPresent( xyzw , "xyzw") << std::endl ;  

    std::cout << SPresent( rotate, "rotate") << std::endl ;  
    std::cout << SPresent( irotate, "irotate") << std::endl ;  

    std::cout << SPresent( scale, "scale") << std::endl ;  
    std::cout << SPresent( iscale, "iscale") << std::endl ;  

    std::cout << SPresent( translate, "translate") << std::endl ;  
    std::cout << SPresent( itranslate, "itranslate") << std::endl ;  
}

template struct SCenterExtentFrame<float> ; 
template struct SCenterExtentFrame<double> ; 


