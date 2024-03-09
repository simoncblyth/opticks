/**
UseGLMQuat.cc
==============

https://web.engr.oregonstate.edu/~mjb/vulkan/Handouts/quaternions.1pp.pdf

::

   ~/o/examples/UseGLMQuat/UseGLMQuat.sh 

**/


#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#include <glm/glm.hpp>
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

template<typename T>
static std::string DescValue(const T t)
{
    std::stringstream ss ; 
    ss << std::fixed << std::setw(10) << std::setprecision(4) << t ; 
    std::string str = ss.str(); 
    return str ; 
}

template<typename T>
static std::string Desc(const T* tt, int num)
{   
    std::stringstream ss ; 
    for(int i=0 ; i < num ; i++) 
        ss  
            << ( i % 4 == 0 && num > 4 ? ".\n" : "" ) 
            << " " << std::fixed << std::setw(10) << std::setprecision(4) << tt[i] 
            << ( i == num-1 && num > 4 ? ".\n" : "" ) 
            ;   

    std::string str = ss.str(); 
    return str ; 
}   

struct angle_axis
{
    float angle ; 
    glm::vec3 axis ; 

    std::string desc() const ; 
};


/**
glm::quat repr of (angle,axis) rotation appears to be::

   quat( cosf(angle/2), sinf(angle/2)*axis ) 

**/

inline std::string angle_axis::desc() const 
{
    std::stringstream ss ; 
    ss
        << " angle         " << DescValue<float>(angle)
        << " sinf(angle)   " << DescValue<float>(sinf(angle)) 
        << " cosf(angle)   " << DescValue<float>(cosf(angle))
        << std::endl
        << " angle/2       " << DescValue<float>(angle/2.f)
        << " sinf(angle/2) " << DescValue<float>(sinf(angle/2.f)) 
        << " cosf(angle/2) " << DescValue<float>(cosf(angle/2.f))
        << std::endl
        << " axis: " << glm::to_string( axis ) 
        << std::endl
        ;
    std::string str = ss.str(); 
    return str ; 
}
//   
//
// +

void test_angle_axis()
{
    //float deg = 0.f ; 
    //float deg = 90.f ; 
    //float deg = 180.f ; 
    //float deg = 270.f ; 
    //float deg = 360.f ; 

    //float deg = 360.f + 90.f ; 
    //float deg = 360.f + 180.f ; 
    float deg = 360.f + 270.f ; 
    //float deg = 360.f + 360.f ; 

    std::vector<angle_axis> vaa ; 
    vaa.push_back( { glm::radians(deg), { 1.f, 0.f, 0.f }} ) ; 
    vaa.push_back( { glm::radians(deg), { 0.f, 1.f, 0.f }} ) ; 
    vaa.push_back( { glm::radians(deg), { 0.f, 0.f, 1.f }} ) ; 
 
    int naa = vaa.size(); 
    for(int i=0 ; i < naa ; i++)
    {

        // 1. form q0 using angleAxis
        const angle_axis& aa = vaa[i] ;    
        glm::quat q0 = glm::angleAxis( aa.angle, aa.axis ) ; 

        // 2. form q1 via ctor with knowledge of quat (angle,axis) formula 
        float c = cosf(aa.angle/2) ; 
        float s = sinf(aa.angle/2) ; 
        glm::quat q1 = glm::quat( c, s*aa.axis.x, s*aa.axis.y, s*aa.axis.z ) ; 

        // 3. cast q0,q1 to mat4 m0,m1 rotation matrices
        glm::mat4 m0 = glm::mat4_cast(q0); 
        glm::mat4 m1 = glm::mat4_cast(q1); 
        //glm::mat4 m1 = glm::toMat4(q0);   // equiv ?

        // 5. cast m0,m1 mat4 back to qm0, qm1 quat 
        glm::quat qm0 = glm::quat_cast(m0) ; 
        glm::quat qm1 = glm::quat_cast(m1) ; 
  
        std::cout << aa.desc() << std::endl ; 
        std::cout << "q0:" << glm::to_string( q0 ) << std::endl ; 
        std::cout << "q1:" << glm::to_string( q1 ) << std::endl ; 
        std::cout << "qm0:" << glm::to_string( qm0 ) << std::endl ; 
        std::cout << "qm1:" << glm::to_string( qm1 ) << std::endl ; 
        std::cout << "-qm0:" << glm::to_string( -qm0 ) << std::endl ; 
        std::cout << "-qm1:" << glm::to_string( -qm1 ) << std::endl ; 

        std::cout << " casting from quat->mat4->quat ... sometimes flips sign (but hypersphere double covers rotations so no problem)" << std::endl ; 

        std::cout << "m0:" << Desc<float>( glm::value_ptr(m0), 16 ) << std::endl ;
        std::cout << "m1:" << Desc<float>( glm::value_ptr(m1), 16 ) << std::endl ;
    }
}


void test_combine_rotations()
{
    glm::quat rot1 = glm::angleAxis( glm::radians(45.f), glm::vec3( 0.707f, 0.707f, 0.f ) );
    glm::quat rot2 = glm::angleAxis( glm::radians(90.f), glm::vec3( 1.f , 0.f , 0.f) );

    std::cout << "rot1:" << glm::to_string( rot1 ) << std::endl ; 
    std::cout << "rot2:" << glm::to_string( rot2 ) << std::endl ; 

    glm::quat rot12 = rot2 * rot1;
    std::cout << "rot12:" << glm::to_string( rot12 ) << std::endl ; 

    glm::vec4 v = glm::vec4( 1., 1., 1., 1. );

    glm::vec4 rot12_v = rot12 * v;

    std::cout << "rot12_v:" << glm::to_string( rot12_v ) << std::endl ; 
    
    glm::mat4 rot12Mat = glm::toMat4( rot12 );

    std::cout << "rot12Mat:" << glm::to_string( rot12Mat ) << std::endl ; 

    glm::vec4 rot12Mat_v = rot12Mat * v; 

    std::cout << "rot12Mat_v:" << glm::to_string( rot12Mat_v ) << std::endl ; 
}

void test_identity()
{
    glm::quat q0(1.f, 0.f, 0.f, 0.f ); 
    glm::mat4 m0 = glm::mat4_cast(q0) ; 

    std::cout << "q0:" << glm::to_string( q0 ) << std::endl ; 
    std::cout << "m0:" << Desc<float>( glm::value_ptr(m0), 16 ) << std::endl ;

    glm::quat q1 = glm::angleAxis( 0.f, glm::vec3( 1.f, 0.f, 0.f ) ); 
    glm::mat4 m1 = glm::mat4_cast(q1) ; 

    std::cout << "q1:" << glm::to_string( q1 ) << std::endl ; 
    std::cout << "m1:" << Desc<float>( glm::value_ptr(m1), 16 ) << std::endl ;


    float a2 = 0.f ;  
    glm::quat q2 = glm::angleAxis( glm::radians(a2), glm::normalize(glm::vec3( 1.f, 1.f, 1.f )) ); 
    glm::mat4 m2 = glm::mat4_cast(q2) ; 
    std::cout << "a2: " << a2 << " (degrees)  [angle 0.f ... identity quat at N pole of hyperphere] " << std::endl ; 
    std::cout << "q2:" << glm::to_string( q2 ) << " any axis is scaled to zero by sin(angle/2) " << std::endl ; 
    std::cout << "m2:" << Desc<float>( glm::value_ptr(m2), 16 ) << std::endl ;

    float a3 = 360.f ;  
    glm::quat q3 = glm::angleAxis( glm::radians(a3), glm::normalize(glm::vec3( 1.f, 1.f, 1.f )) ); 
    glm::mat4 m3 = glm::mat4_cast(q3) ; 
    std::cout << "a3: " << a3 << " (degrees) [angle 360 .. conjugate identity at S pole of hyperphere ]" << std::endl ; 
    std::cout << "q3:" << glm::to_string( q3 ) << " any axis is scaled to zero by sin(angle/2) " << std::endl ; 
    std::cout << "m3:" << Desc<float>( glm::value_ptr(m3), 16 ) << std::endl ;
}


void test_rotate_between_vectors_0()
{
    glm::vec3 a(1.f, 0.f, 0.f ); 
    glm::vec3 b(0.f, 1.f, 0.f ); 
 
    glm::quat q( glm::dot(a,b), glm::cross(a,b) ) ; 
    glm::quat r = glm::rotation(a, b); 
    glm::quat rn = glm::normalize(r); 

    glm::mat4 m = glm::mat4_cast(q); 

    std::cout << "q " <<  glm::to_string( q ) << std::endl ;
    std::cout << "r " <<  glm::to_string( r ) << std::endl ;
    std::cout << "rn " <<  glm::to_string( rn ) << std::endl ;

    std::cout << "m " << Desc<float>( glm::value_ptr(m), 16 ) << std::endl ;
}

void test_rotate_between_vectors_1()
{
    //glm::vec3 _a(1.f, 0.f, 0.f ); 
    //glm::vec3 _b(0.f, 1.f, 0.f ); 

    glm::vec3 _a(1.f, 1.f, 0.f ); 
    glm::vec3 _b(0.f, 1.f, 1.f ); 

    glm::vec3 a = glm::normalize(_a); 
    glm::vec3 b = glm::normalize(_b); 

    glm::quat qa = glm::quat( 0.f, a ); 
    glm::quat qb = glm::quat( 0.f, b ); 

    // Shoemake starts from q0=0 pure quats, claiming the a->b q just
    glm::quat q0( glm::dot(a,b), glm::cross(a,b) ) ;  
    glm::mat4 m0 = glm::mat4_cast(q0); 

    glm::quat q1 = qb*glm::conjugate(qa) ; 
    glm::mat4 m1 = glm::mat4_cast(q1); 

    glm::quat q2 = qa*glm::conjugate(qb) ; 

    std::cout << " a " << glm::to_string( a ) << std::endl ;  
    std::cout << " b " << glm::to_string( b ) << std::endl ;  
    std::cout << "qa " <<  glm::to_string( qa ) << " glm::quat( 0.f, a ) : pure quat (equator of hypersphere) " << std::endl ;
    std::cout << "qb " <<  glm::to_string( qb ) << " glm::quat( 0.f, b ) : pure quat (equator of hypersphere) " << std::endl ;
    std::cout << "q0 " <<  glm::to_string( q0 ) << " q0( glm::dot(a,b), glm::cross(a,b) ) : valid for pure q? Ken Shoemake " << std::endl ;
    std::cout << "q1 " <<  glm::to_string( q1 ) << " qb*glm::conjugate(qa) " << std::endl ;

    //std::cout << "q2 " <<  glm::to_string( q2 ) << " qa*glm::conjugate(qb) " << std::endl ;


    std::cout << "m0 " << Desc<float>( glm::value_ptr(m0), 16 ) << std::endl ;
    std::cout << "m1 " << Desc<float>( glm::value_ptr(m1), 16 ) << std::endl ;



}



int main()
{
    //test_identity(); 
    //test_angle_axis(); 
    //test_combine_rotations(); 

    //test_rotate_between_vectors_0(); 
    test_rotate_between_vectors_1(); 

    return 0 ; 
}
