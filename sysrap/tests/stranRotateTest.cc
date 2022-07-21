#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"

#include "SPath.hh"
#include "NP.hh"


#include <glm/gtc/random.hpp>
#include <glm/gtx/component_wise.hpp>


template<typename T>
struct stranRotateTest
{
    static const T MAX_DEVIATION ; 
    static const char* NAME ; 

    static T make_rotate_a2b_(const glm::tvec3<T>& a_, const glm::tvec3<T>& b_, bool dump ); 

    static void make_rotate_a2b_0(unsigned num); 
    static void make_rotate_a2b_1(); 
    static void make_rotate_a2b_2(); 

    static T make_rotate_a2b_3(unsigned num); 
    static T make_rotate_a2b_4(unsigned num); 

    static void test_a2b(); 

};

template<>
const float stranRotateTest<float>::MAX_DEVIATION = 1e-6f ; 

template<>
const double stranRotateTest<double>::MAX_DEVIATION = 1e-12f ; 


template<>
const char* stranRotateTest<float>::NAME = "stranRotateTest<float>" ; 

template<>
const char* stranRotateTest<double>::NAME = "stranRotateTest<double>" ; 




template<typename T>
T stranRotateTest<T>::make_rotate_a2b_(const glm::tvec3<T>& a_, const glm::tvec3<T>& b_, bool dump )
{
    glm::tvec3<T> a = glm::normalize(a_); 
    glm::tvec3<T> b = glm::normalize(b_); 
    bool flip = true ; 

    const Tran<T>* tr = Tran<T>::make_rotate_a2b( a, b, flip ); 

    T w = 0. ; 
    glm::tvec4<T> a4(a.x,a.y,a.z,w); 

    glm::tvec4<T> b4 = tr->t * a4 ;
    glm::tvec3<T> b3(b4); 
    T dmax = glm::compMax( glm::abs( b3 - b ) ); 

    glm::tvec4<T> B4 = a4 * tr->t ;
    glm::tvec3<T> B3(B4); 
    T Dmax = glm::compMax( glm::abs( B3 - b ) ); 

    if(dump)
    {
        std::cout 
            << "make_rotate_a2b_ " 
            << std::endl  
            << " a " << glm::to_string(a) 
            << std::endl 
            << " b " << glm::to_string(b) 
            << std::endl 
            << " b4 " << glm::to_string(b4) 
            << std::endl 
            << " b3 " << glm::to_string(b3) 
            << std::endl 
            << " B3 " << glm::to_string(B3) 
            << std::endl 
            << " dmax*1e6 " << std::setw(10) << std::fixed << std::setprecision(10) << dmax*1e6 
            << std::endl
            << " Dmax*1e6 " << std::setw(10) << std::fixed << std::setprecision(10) << Dmax*1e6 
            << std::endl


            ; 

        std::cout << *tr << std::endl ;   

        for(int i=0 ; i < 3 ; i++) std::cout 
                << std::setw(1) << i << " : " 
                << std::setw(10) << std::fixed << std::setprecision(10) << ( b[i] - b4[i] ) 
                << std::endl 
                ;
    }
 


    return dmax ; 
}
 
 

template<typename T>
void stranRotateTest<T>::make_rotate_a2b_0(unsigned num)
{
    glm::tvec3<T> px(1,0,0) ; 
    glm::tvec3<T> py(0,1,0) ; 
    glm::tvec3<T> pz(0,0,1) ; 

    glm::tvec3<T> nx(-1, 0, 0) ; 
    glm::tvec3<T> ny( 0,-1, 0) ; 
    glm::tvec3<T> nz( 0, 0,-1) ; 

    if(num > 0)  make_rotate_a2b_(px, py, true); 
    if(num > 1 ) make_rotate_a2b_(pz, pz, true); 
    if(num > 2 ) make_rotate_a2b_(pz, nz, true); 
}


template<typename T>
void stranRotateTest<T>::make_rotate_a2b_1()
{
     glm::tvec3<T> a(1,0,0) ; 

     for(T p=0. ; p < 2. ; p+=0.1 )
     {
         T phi = glm::pi<T>()*p  ;

         glm::tvec3<T> b(cos(phi),sin(phi),0) ; 

         std::cout << " a " << glm::to_string(a) << std::endl ; 
         std::cout << " b " << glm::to_string(b) << std::endl ; 
 
         const Tran<T>* t = Tran<T>::make_rotate_a2b( a, b ); 

         std::cout << *t << std::endl ;   
     }
}

template<typename T>
void stranRotateTest<T>::make_rotate_a2b_2()
{
    glm::tvec3<T> a( 1, 1,0) ; 
    glm::tvec3<T> b(-1,-1,0) ; 
    make_rotate_a2b_(a, b, true); 
}



/**
stranRotateTest::make_rotate_a2b_3
------------------------

0. Generate two random unit vectors a,b 
1. compute a rotation matrix R that transforms a->b 
2. use R to transform a 
3. compare R a with b    
 
**/

template<typename T>
T stranRotateTest<T>::make_rotate_a2b_3(unsigned num)
{
    bool dump = num < 3 ;
    //bool dump = false ;
    T dmaxx = 0. ; 
 
    for(unsigned i=0 ; i < num ; i++)
    {
        glm::tvec3<T> a = glm::sphericalRand(1.); 
        glm::tvec3<T> b = glm::sphericalRand(1.); 

        T dmax = make_rotate_a2b_(a,b, dump); 
        if( dmax > dmaxx ) dmaxx = dmax ; 

        if(dump) std::cout 
            << " i " << std::setw(4) << i 
            << " a " << std::setw(40) << glm::to_string(a) 
            << " b " << std::setw(40) << glm::to_string(b) 
            << " dmax*1e9 " << std::fixed << std::setw(10) << std::setprecision(6) << dmax*1e9
            << std::endl
            ; 
    }
    std::cout 
        << " test_make_rotate_a2b_3 "
        << " dmaxx " << std::scientific << dmaxx
        << " dmaxx*1e9 " << std::fixed << std::setw(10) << std::setprecision(6) << dmaxx*1e9
        << std::endl
        ; 
    return dmaxx ; 
}


/**
make_rotate_a2b_4 : checking handling of parallel or anti-parallel
--------------------------------------------------------------------------

0. Generate random unit vectors a and obtain b from it by negation 
1. compute a rotation matrix R that transforms a->b 
2. use R to transform a 
3. compare R a with b    
 
**/

template<typename T>
T stranRotateTest<T>::make_rotate_a2b_4(unsigned num )
{
    bool dump = num < 3 ;
    //bool dump = false ;

    T dmaxx = 0. ; 
 
    for(unsigned i=0 ; i < num ; i++)
    {
        glm::tvec3<T> a = glm::sphericalRand(1.); 
        glm::tvec3<T> b = i % 2 == 0 ? a : -a ; 

        T dmax = make_rotate_a2b_(a,b, dump); 
        if( dmax > dmaxx ) dmaxx = dmax ; 

        if(dump) std::cout 
            << " i " << std::setw(4) << i 
            << " a " << std::setw(40) << glm::to_string(a) 
            << " b " << std::setw(40) << glm::to_string(b) 
            << " dmax*1e9 " << std::fixed << std::setw(10) << std::setprecision(6) << dmax*1e9
            << std::endl
            ; 
    }

    std::cout 
        << " test_make_rotate_a2b_4 "
        << " dmaxx " << std::scientific << dmaxx
        << " dmaxx*1e9 " << std::fixed << std::setw(10) << std::setprecision(6) << dmaxx*1e9
        << std::endl
        ; 

    return dmaxx ; 
}


template<typename T>
void stranRotateTest<T>::test_a2b()
{
    std::cout << NAME << std::endl ; 
    //test_make_rotate_a2b_0<T>(1u); 

    T d3 = make_rotate_a2b_3(1000); 
    assert( d3 < MAX_DEVIATION ); 

    T d4 = make_rotate_a2b_4(1000); 
    assert( d4 < MAX_DEVIATION ); 
}


int main()
{
    stranRotateTest<double>::test_a2b(); 
    stranRotateTest<float>::test_a2b(); 

    return 0 ; 
}
// om- ; TEST=stranRotateTest om-t



