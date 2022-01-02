
#include <cstdlib>
#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>


template<typename T>
struct Tangential
{
    static const T pi ; 
    static void GetEVector( std::vector<T>& vec, const char* ekey, const char* fallback );
    static glm::tmat4x4<T> Rotation( T theta, T phi ) ; 
    static glm::tmat4x4<T> IRotation( T theta, T phi ) ; 
    static glm::tmat4x4<T> Translation( T radius,  T theta, T phi ) ; 
    static glm::tmat4x4<T> ITranslation( T radius, T theta, T phi ) ; 

    static void CartesianToSpherical( glm::tvec3<T>& radius_theta_phi, const glm::tvec4<T>& xyzw ); 

    Tangential( T radius, T theta, T phi ); 

    void conventional_to_tangential( glm::tvec4<T>& rtpw , const glm::tvec4<T>& xyzw ); 
    void tangential_to_conventional( glm::tvec4<T>& xyzw , const glm::tvec4<T>& rtpw ); 
    void dump() const ;  

    glm::tmat4x4<T> rot ; 
    glm::tmat4x4<T> iro ; 
    glm::tmat4x4<T> tra ; 
    glm::tmat4x4<T> itr ; 

    glm::tmat4x4<T> rot_itr ; 
    glm::tmat4x4<T> tra_iro ; 
}; 

template<typename T>
const T Tangential<T>::pi = glm::pi<T>() ; 


template<typename T>
void Tangential<T>::CartesianToSpherical( glm::tvec3<T>& radius_theta_phi, const glm::tvec4<T>& xyzw )
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
Tangential::Rotation
-----------------------

Spherical unit vectors (ru tu pu) related to cartesian unit vectors (xu yu zu)
via orthogonal rotation matrix::

      | ru |     |   sin(theta)cos(phi)    sin(theta)sin(phi)      cos(theta)   |  | xu | 
      |    |     |                                                              |  |    |
      | tu | =   |  cos(theta)cos(phi)    cos(theta)sin(phi)     -sin(theta)    |  | yu |
      |    |     |                                                              |  |    |
      | pu |     |  -sin(phi)                 cos(phi)              0           |  | zu |

**/

template<typename T>
glm::tmat4x4<T> Tangential<T>::Rotation( T theta, T phi )
{
    std::array<T, 16> _rot = {{
        sin(theta)*cos(phi),   sin(theta)*sin(phi),   cos(theta),  0. ,
        cos(theta)*cos(phi),   cos(theta)*sin(phi),  -sin(theta),  0. ,
        -sin(phi),             cos(phi),              0.,          0. ,
         0.,                   0.,                    0.,          1.
      }} ;  
    return glm::make_mat4x4<T>(_rot.data()) ; 
}

/**
Tangential::IRotation
-----------------------

Cartesian unit vectors (xu yu zu) in terms of spherical unit vectors related by 
the inverse of the above transform which is its transpose::


      | xu |     |   sin(theta)cos(phi)    cos(theta)cos(phi)      -sin(phi)    |  | ru | 
      |    |     |                                                              |  |    |
      | yu | =   |  sin(theta)sin(phi)    cos(theta)sin(phi)        cos(phi)    |  | tu |
      |    |     |                                                              |  |    |
      | zu |     |   cos(theta)               -sin(theta)              0        |  | pu |

**/

template<typename T>
glm::tmat4x4<T> Tangential<T>::IRotation(  T theta, T phi )
{
    std::array<T, 16> _iro = {{
        sin(theta)*cos(phi),   cos(theta)*cos(phi),  -sin(phi),     0. ,
        sin(theta)*sin(phi),   cos(theta)*sin(phi),  cos(phi),      0. ,
        cos(theta),            -sin(theta),          0.,            0. ,
         0.,                   0.,                   0.,            1.
      }} ;  
    return glm::make_mat4x4<T>(_iro.data()) ; 
}

template<typename T>
glm::tmat4x4<T> Tangential<T>::Translation( T radius, T theta, T phi )
{
    std::array<T, 16> _tra = {{
         1.,                           0.,                          0.,                  0. ,
         0.,                           1.,                          0.,                  0. ,
         0.,                           0.,                          1.,                  0. ,
         radius*sin(theta)*cos(phi),   radius*sin(theta)*sin(phi),  radius*cos(theta),   1. 
      }} ;  
    return glm::make_mat4x4<T>(_tra.data()) ; 
}

template<typename T>
glm::tmat4x4<T> Tangential<T>::ITranslation( T radius, T theta, T phi )
{
    std::array<T, 16> _itr = {{
         1.,                           0.,                          0.,                  0. ,
         0.,                           1.,                          0.,                  0. ,
         0.,                           0.,                          1.,                  0. ,
         -radius*sin(theta)*cos(phi),  -radius*sin(theta)*sin(phi), -radius*cos(theta),  1. 
      }} ;  
    return glm::make_mat4x4<T>(_itr.data()) ; 
}


template<typename T>
Tangential<T>::Tangential(T radius, T theta, T phi )
    :
    rot(Rotation(pi*theta,pi*phi)),
    iro(IRotation(pi*theta,pi*phi)),
    tra(Translation(radius,pi*theta,pi*phi)),
    itr(ITranslation(radius,pi*theta,pi*phi)),
    rot_itr(rot * itr),
    tra_iro(tra * iro)
{
}


template<typename T>
void Tangential<T>::tangential_to_conventional( glm::tvec4<T>& xyzw , const glm::tvec4<T>& rtpw )
{
    //xyzw = rot * itr * rtpw ; 
    xyzw = rot_itr * rtpw ; 
}

template<typename T>
void Tangential<T>::conventional_to_tangential( glm::tvec4<T>& rtpw , const glm::tvec4<T>& xyzw )
{
    //rtpw = tra * iro * xyzw ; 
    rtpw = tra_iro * xyzw ; 
}


template<typename T>
void Tangential<T>::dump() const 
{
    std::cout  << " rot " << glm::to_string(rot) << std::endl ; 
    std::cout  << " iro " << glm::to_string(iro) << std::endl ; 
    std::cout  << " tra " << glm::to_string(tra) << std::endl ; 
    std::cout  << " itr " << glm::to_string(itr) << std::endl ; 
}

template<typename T>
void Tangential<T>::GetEVector( std::vector<T>& vec, const char* ekey, const char* fallback )  // static
{
    const char* sval = getenv(ekey); 
    std::stringstream ss(sval ? sval : fallback); 
    std::string s ; 
    while(getline(ss, s, ',')) 
    {
        T val(0) ; 
        std::stringstream vv(s.c_str()); 
        vv >> val ; 
        vec.push_back(val); 
    }
}  

int main(int argc, char** argv)
{
    std::vector<double> rtp ;  
    Tangential<double>::GetEVector(rtp, "RTP", "20,0.25,0.0") ;  
    for(unsigned i=0 ; i < rtp.size() ; i++) std::cout << std::setw(4) << i << " : " << rtp[i] << std::endl ; 
    assert( rtp.size() == 3 ); 
    double radius = rtp[0] ; 
    double theta  = rtp[1] ; 
    double phi    = rtp[2] ; 

    Tangential<double> ta(radius, theta, phi ); 
    ta.dump();

    glm::tvec4<double> rtpw(0., 0., 0., 1.) ; 
    glm::tvec4<double> xyzw(0., 0., 0., 1.) ; 
    glm::tvec4<double> rtpw2(0., 0., 0., 1.) ; 

    for(int t=-3 ; t <= 3 ; t++)
    for(int p=-3 ; p <= 3 ; p++)
    {
        rtpw.y = double(t) ;    
        rtpw.z = double(p) ;    

        ta.tangential_to_conventional(xyzw, rtpw ); 
        ta.conventional_to_tangential(rtpw2, xyzw ); 

        std::cout 
            << "rtpw" 
            << "(" << std::setw(10) << std::fixed << std::setprecision(4) << rtpw.x
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << rtpw.y
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << rtpw.z
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << rtpw.w
            << ")"
            << " xyzw " 
            << "(" << std::setw(10) << std::fixed << std::setprecision(4) << xyzw.x
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << xyzw.y
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << xyzw.z
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << xyzw.w
            << ")"
            << "rtpw2" 
            << "(" << std::setw(10) << std::fixed << std::setprecision(4) << rtpw2.x
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << rtpw2.y
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << rtpw2.z
            << " " << std::setw(10) << std::fixed << std::setprecision(4) << rtpw2.w
            << ")"
            << std::endl 
            ;
    }

    return 0 ; 
}
