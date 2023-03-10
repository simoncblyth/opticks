// ./stra_test.sh

#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "stra.h"


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

int main()
{
    test_Place(); 

    return 0 ; 
}
