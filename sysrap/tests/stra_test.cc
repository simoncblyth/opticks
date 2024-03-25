// ~/o/sysrap/tests/stra_test.sh

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

template<typename T>
void test_Transform_Vec()
{
    std::cout << "test_Transform_Vec" << std::endl ; 

    std::array<T,16> a = { 
          1., 0., 0., 0., 
          0., 1., 0., 0., 
          0., 0., 1., 0., 
          0., 0.,100.,1. 
     }; 

    glm::tmat4x4<T> m = stra<T>::FromData(a.data()) ;   
    std::cout << " m " << m << std::endl ; 

    glm::tvec4<T> pos ; 
    glm::tvec4<T> pos0 = { 0, 0, 0, 1 } ;  

    stra<T>::Transform_Vec(pos, pos0, m ); 

    std::cout << " pos0 " << stra<T>::Desc(pos0)  << std::endl ; 
    std::cout << " pos  " << stra<T>::Desc(pos)  << std::endl ; 

}




template<typename T>
void test_Transform_Data()
{
    std::cout << "test_Transform_Data" << std::endl ; 

    std::array<T,16> a = { 
          1., 0., 0., 0., 
          0., 1., 0., 0., 
          0., 0., 1., 0., 
          0., 0.,100.,1. 
     }; 

    glm::tmat4x4<T> m = stra<T>::FromData(a.data()) ;   
    std::cout << " m " << m << std::endl ; 

    std::array<double,3> pos0 = { 0, 0, 0 } ; 
    std::array<double,3> pos = { 0, 0, 0 } ; 

    stra<T>::Transform_Data(pos.data(), pos0.data(), m ); 

    std::cout << " pos0 " << stra<T>::Desc(pos0.data(), 1, 3, 0)  << std::endl ; 
    std::cout << " pos  " << stra<T>::Desc(pos.data(),  1, 3, 0 ) << std::endl ; 

}

template<typename T>
void test_Desc_strided()
{
    std::array<T,64> a = { 
          1., 2., 3., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 

          4., 5., 6., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 

          7., 8., 9., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 

         10.,11.,12., 0.,
          0., 0., 0., 0.,
          0., 0., 0., 0., 
          0., 0., 0., 0. 
     }; 


   std::cout 
       << "test_Desc_strided"
       << std::endl 
       << stra<T>::Desc( a.data(), 4, 3, 16 ) 
       << std::endl 
       ;
              
}





template<typename T>
void test_Transform_Strided()
{
    std::cout << "test_Transform_Strided" << std::endl ; 

    std::array<T,16> _m = { 
          1., 0., 0., 0., 
          0., 1., 0., 0., 
          0., 0., 1., 0., 
          0., 0.,100.,1. 
     }; 

    glm::tmat4x4<T> m = stra<T>::FromData(_m.data()) ;   
    std::cout << " m " << m << std::endl ; 


    std::array<T,64> p0 = { 
          1., 2., 3., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 

          4., 5., 6., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 

          7., 8., 9., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 
          0., 0., 0., 0., 

         10.,11.,12., 0.,
          0., 0., 0., 0.,
          0., 0., 0., 0., 
          0., 0., 0., 0. 
     }; 

     std::array<T,64> p1(p0) ; 

     stra<T>::Transform_Strided( p1.data(), p0.data(), 4, 3, 16, m, 1. ); 

     std::array<T,64> p2(p0) ; 
     stra<T>::Transform_Strided_Inplace( p2.data(), 4, 3, 16, m, 1. ); 


     std::cout 
         << "test_Transform_Strided"
         << std::endl 
         << " p0 "
         << std::endl 
         << stra<T>::Desc( p0.data(), 4, 3, 16 ) 
         << std::endl 
         << " p1 "
         << std::endl 
         << stra<T>::Desc( p1.data(), 4, 3, 16 ) 
         << std::endl 
         << " p2 (check Transform_Strided_Inplace)  "
         << std::endl 
         << stra<T>::Desc( p2.data(), 4, 3, 16 ) 
         << std::endl 
         ;

}





template<typename T>
void test_MakeTransformedArray()
{
    std::cout << "test_MakeTransformedArray" << std::endl ; 

    std::array<double,12> _aa = {{
          0., 0., 0., 
          1., 0., 0., 
          0., 1., 0., 
          0., 0., 1.  
    }}; 

    int itemsize = sizeof(double)*3 ;
    NP* a = NP::Make<double>(4, 3 ); 
    a->read2( _aa.data() ); 
    const double* aa = a->cvalues<double>(); 

    std::cout << "a" << std::endl ; 
    for(int i=0 ; i < a->shape[0] ; i++)
    {
        glm::tvec3<double> v(0.) ; 
        memcpy( glm::value_ptr(v) , aa + i*3, itemsize ); 
        std::cout << glm::to_string(v) << std::endl ;   
    }


    glm::tmat4x4<double> t = {
         { 1., 0., 0., 0. },
         { 0., 1., 0., 0. },
         { 0., 0., 1., 0. },
         { 0., 0.,10., 1. } } ; 

    NP* b = stra<double>::MakeTransformedArray(a, &t ); 
    const double* bb = b->cvalues<double>(); 

    std::cout << "b" << std::endl ; 
    for(int i=0 ; i < b->shape[0] ; i++)
    {
        glm::tvec3<double> v(0.) ; 
        memcpy( glm::value_ptr(v) , bb + i*3, itemsize ); 
        std::cout << glm::to_string(v) << std::endl ;   
    }
}


template<typename T>
void test_Copy_Columns_3x4()
{
    std::cout << "test_Copy_Columns_3x4" << std::endl ; 
    std::array<T,16> aa = {{
          0., 1., 2., 3., 
          4., 5., 6., 7.,
          8., 9.,10.,11., 
         12.,13.,14.,15.  
    }}; 

    glm::tmat4x4<T> src(0.) ; 
    glm::tmat4x4<T> dst(0.) ;
 
    memcpy( glm::value_ptr(src) , aa.data(), 16*sizeof(T) ); 
    stra<T>::Copy_Columns_3x4(dst, src); 


    std::cout
        << " src \n"  
        << glm::to_string(src) 
        << std::endl 
        << stra<T>::Desc(src)
        << std::endl 
        ; 

    std::cout
        << " dst \n"  
        << glm::to_string(dst) 
        << std::endl 
        << stra<T>::Desc(dst)
        << std::endl 
        ; 

}


int main()
{
    /*
    test_Place(); 
    test_Rows<double>();
    test_Transform_AABB<double>();
    test_Transform_AABB_Inplace<double>();
    test_Desc_strided<double>();
    test_Transform_Vec<double>();
    test_Transform_Data<double>();
    test_Transform_Strided<double>();
    test_MakeTransformedArray<double>();
    */

    test_Copy_Columns_3x4<double>();


    return 0 ; 
}
