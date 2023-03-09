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
    glm::tvec3<double> vc(-250,0,0) ;   // +X

    bool flip = false ;  
    glm::tmat4x4<double> tr = stra<double>::Place(va, vb, vc, flip ); 
    std::cout << stra<double>::Desc(tr) << std::endl ;  

    const int N = 4 ; 

    std::vector<std::string> alabel(N) ; 
    alabel[0] = "a0 +X" ; 
    alabel[1] = "a1 +Y" ; 
    alabel[2] = "a2 +Z" ; 
    alabel[3] = "a3  O" ; 

    std::vector<glm::tvec4<double>> a(N) ; 
    a[0] = {1,0,0,1 } ;  
    a[1] = {0,1,0,1 } ;
    a[2] = {0,0,1,1 } ;
    a[3] = {0,0,0,1 } ;

    std::vector<std::string> blabel(N) ; 
    blabel[0] = "b0 = a2b * a0 " ; 
    blabel[1] = "b1 = a2b * a1 " ; 
    blabel[2] = "b2 = a2b * a2 " ; 
    blabel[3] = "b3 = a2b * a3 " ; 

    std::vector<glm::tvec4<double>> b(4) ; 
    for(int i=0 ; i < N ; i++) b[i] = tr * a[i] ;     

    for(int i=0 ; i < N ; i++) 
    {
        std::cout 
            << std::setw(15) << alabel[i] 
            << " "
            << stra<double>::Desc(a[i]) 
            << std::endl 
            << std::setw(15) << blabel[i] 
            << " "
            << stra<double>::Desc(b[i]) 
            << std::endl 
            << std::endl 
            ;
    }



}

int main()
{
    test_Place(); 

    return 0 ; 
}
