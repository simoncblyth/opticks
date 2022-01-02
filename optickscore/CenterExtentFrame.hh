#pragma once
/**
CenterExtentFrame
===================

See also 

* ana/tangential.py
* ana/pvprim.py
* ana/pvprim1.py
* ana/symtran.py
* ana/tangential.cc

**/
#include <glm/glm.hpp>
#include "OKCORE_API_EXPORT.hh"

template<typename T>
struct OKCORE_API CenterExtentFrame 
{
    // convert between coordinate systems
    static void CartesianToSpherical( glm::tvec3<T>& radius_theta_phi, const glm::tvec4<T>& xyzw ); 
    static void SphericalToCartesian( glm::tvec4<T>& xyzw, const glm::tvec3<T>& radius_theta_phi );

    // rotation matrices between conventional XYZ and tangential RPT cartesian frames 
    static glm::tmat4x4<T> XYZ_to_RTP( T theta, T phi );
    static glm::tmat4x4<T> RTP_to_XYZ( T theta, T phi );

    CenterExtentFrame( float  _cx, float  _cy, float  _cz, float  _extent, bool rtp_tangential ) ; 
    CenterExtentFrame( double _cx, double _cy, double _cz, double _extent, bool rtp_tangential ) ; 

    void init();  
    void dump(const char* msg="CenterExtentFrame::dump") const ; 

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
}; 


