// ./stra_test.sh

#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "stra.h"
#include "sbb.h"


void test_Desc()
{
    glm::tmat4x4<double> a(1.); 
    glm::tmat4x4<double> b(2.); 
    glm::tmat4x4<double> c(3.); 

    std::cout << stra<double>::Desc(a, b, c, "a", "b", "c" ); 
}

void test_Place()
{
    glm::tvec3<double> va(0,0,1) ;   // +Z
    glm::tvec3<double> vb(1,0,0) ;   // +X
    glm::tvec3<double> vc(-250,0,0) ;  

    bool flip = true ;  
    glm::tmat4x4<double> tr = stra<double>::Place(va, vb, vc, flip ); 
    std::cout << stra<double>::Desc(tr) << std::endl ;  
    std::cout << stra<double>::Array(tr) << std::endl << std::endl ;  

    const int N = 7 ; 
    double sx = 254. ;  
    double sy = 254. ;  
    double sz = 186. ;  

    std::vector<std::string> l(N) ; 
    std::vector<std::string> m(N) ; 
    std::vector<glm::tvec4<double>> a(N) ; 
    std::vector<glm::tvec4<double>> b(N) ; 

    a[0] = {0,0,0,1 }  ; l[0] = "O" ; 

    a[1] = {sx,0,0,1 }  ; l[1] = "+sx" ;  
    a[2] = {0,sy,0,1 }  ; l[2] = "+sy" ; 
    a[3] = {0,0,sz,1 }  ; l[3] = "+sz" ; 

    a[4] = {-sx,0,0,1 } ; l[4] = "-sx" ;  
    a[5] = {0,-sy,0,1 } ; l[5] = "-sy" ; 
    a[6] = {0,0,-sz,1 } ; l[6] = "-sz" ; 


    for(int i=0 ; i < N ; i++) b[i] = tr * a[i] ;     
    for(int i=0 ; i < N ; i++) m[i] = "(tr * " + l[i] + ")" ;     

    for(int i=0 ; i < N ; i++) std::cout 
        << std::setw(15) << l[i] << " " 
        << stra<double>::Desc(a[i]) 
        << std::setw(15) << m[i] << " " 
        << stra<double>::Desc(b[i]) 
        << std::endl 
        ;
}


template<typename T>
void test_Rows()
{
    std::array<T,16> a = { 
          0., 1., 2., 3., 
          4., 5., 6., 7., 
          8., 9.,10.,11., 
         12.,13.,14.,15.
     }; 

    glm::tmat4x4<T> m = stra<T>::FromData(a.data()) ;   
    std::cout << " m " << m << std::endl ; 
 
    glm::tvec4<T> q0(0.);
    glm::tvec4<T> q1(0.);
    glm::tvec4<T> q2(0.);
    glm::tvec4<T> q3(0.);

    stra<T>::Rows(q0,q1,q2,q3,m); 

    std::cout << " q0 " << q0 << std::endl ; 
    std::cout << " q1 " << q1 << std::endl ; 
    std::cout << " q2 " << q2 << std::endl ; 
    std::cout << " q3 " << q3 << std::endl ; 

    glm::tvec4<T> q01_min = glm::min<T>( q0, q1 ); 
    glm::tvec4<T> q01_max = glm::max<T>( q0, q1 ); 

    std::cout << " q01_min " << q01_min << std::endl ; 
    std::cout << " q01_max " << q01_max << std::endl ; 
}


template<typename T>
void test_Transform_AABB()
{
    std::cout << "test_Transform_AABB" << std::endl ; 

    std::array<T,16> a = { 
          1., 0., 0., 0., 
          0., 1., 0., 0., 
          0., 0., 1., 0., 
          0., 0.,100.,1. 
     }; 

    glm::tmat4x4<T> m = stra<T>::FromData(a.data()) ;   
    std::cout << " m " << m << std::endl ; 


    std::array<T,6> bb0 = { -100., -100., -100.,  100., 100., 100. } ; 
    std::array<T,6> bb1(bb0) ; 

    std::cout << " bb0 " << sbb::Desc(bb0.data()) << std::endl ; 

    stra<T>::Transform_AABB( bb1.data(), bb0.data(),  m );   

    std::cout << " bb1 " << sbb::Desc(bb1.data()) << std::endl ; 
}

template<typename T>
void test_Transform_AABB_Inplace()
{
    std::cout << "test_Transform_AABB_Inplace" << std::endl ; 

    std::array<T,16> a = { 
          1., 0., 0., 0., 
          0., 1., 0., 0., 
          0., 0., 1., 0., 
          0., 0.,100.,1. 
     }; 

    glm::tmat4x4<T> m = stra<T>::FromData(a.data()) ;   
    std::cout << " m " << m << std::endl ; 

    std::array<T,6> bb0 = { -100., -100., -100.,  100., 100., 100. } ; 
    std::array<T,6> bb1(bb0) ; 

    std::cout << " bb0 " << sbb::Desc(bb0.data()) << std::endl ; 

    stra<T>::Transform_AABB_Inplace( bb1.data(),  m );   

    std::cout << " bb1 " << sbb::Desc(bb1.data()) << std::endl ; 
}

int main()
{
    /*
    test_Place(); 
    test_Rows<double>();
    test_Transform_AABB<double>();
    */
    test_Transform_AABB_Inplace<double>();

    return 0 ; 
}
