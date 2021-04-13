#include <cassert>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>


void test_mul()
{
    glm::vec2 a(1.f, 2.f ); 
    glm::vec2 b(10.f, 20.f ); 
    glm::vec2 ab_expect(10.f, 40.f);  

    glm::vec2 ab = a*b ; 
    std::cout 
        << " ab.x " << ab.x 
        << " ab.y " << ab.y
        << std::endl 
        ; 

    assert( ab_expect.x == ab.x ); 
    assert( ab_expect.y == ab.y ); 
}


void test_div()
{
    glm::vec2 p(100.f, 150.f); 
    glm::vec2 ab(10.f, 15.f ); 
    glm::vec2 poab = p/ab ; 
    glm::vec2 poab_expect(10.f, 10.f) ; 

    std::cout 
        << " poab.x " << poab.x 
        << " poab.y " << poab.y
        << std::endl 
        ; 

    assert( poab_expect.x == poab.x ); 
    assert( poab_expect.y == poab.y ); 
} 

void test_dot()
{


    glm::vec2 ab(100.f, 150.f); 
    std::cout << " ab (" << ab.x << "," << ab.y << ")" << std::endl ; 

    std::vector<glm::vec2> pp ; 
    pp.push_back( {100.f,   0.f } );
    pp.push_back( {  0.f, 150.f } );
    pp.push_back( {  0.f,   0.f } );
    pp.push_back( {100.f, 150.f } );

    for(int i=0 ; i < int(pp.size()) ; i++)
    { 
        const glm::vec2& p = pp[i] ;  
        float d = glm::dot( p/ab, p/ab ); 
        std::cout << " p( " << p.x << "," << p.y << ") " << d << std::endl ;    
    }
}




int main(int argc, char** argv)
{
#ifdef GLM_HAS_TEMPLATE_ALIASES
    std::cout << "GLM_HAS_TEMPLATE_ALIASES" << std::endl ; 
#else
    std::cout << "OOPS NOT DEFINED : GLM_HAS_TEMPLATE_ALIASES" << std::endl ; 
#endif

    glm::vec4 v(1.f, 2.f, 3.f, 4.f); 

    for(int i=0 ; i < 4 ; i++ ) std::cout << v[i] << " " ; 
    std::cout << std::endl ; 
    assert( v.w == 4.f ); 


    glm::tmat4x4<double> m(1.); 

    for(int i=0 ; i < 4 ; i++ ) for(int j=0 ; j < 4 ; j++) std::cout << m[i][j] << " " ; 
    std::cout << std::endl ; 
    assert( m[3][3] == 1. ); 

    test_mul(); 
    test_div(); 
    test_dot(); 



    return 0 ; 
}
