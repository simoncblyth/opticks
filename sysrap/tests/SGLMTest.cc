#include <cassert>

#include <glm/glm.hpp>
#include "OPTICKS_LOG.hh"
#include "SGLM.h"


void test_GetEVec()
{
    glm::vec3 v(0.f, 0.f, 0.f ); 
    SGLM::GetEVec(v, "VEC3" , "1,2,3" );  

    assert( v.x == 1.f ); 
    assert( v.y == 2.f ); 
    assert( v.z == 3.f ); 

    const char* key = "VEC4" ; 

    glm::vec4 f(SGLM::EVec4(key, "10,20,30,40"));

    LOG(info) << std::setw(10) << key << SGLM::Present(f) ; 
   
    assert( f.x == 10.f ); 
    assert( f.y == 20.f ); 
    assert( f.z == 30.f ); 
    assert( f.w == 40.f ); 
}


void test_Narrow()
{
    glm::tmat4x4<double> md = SGLM::DemoMatrix<double>(1.); 
    glm::tmat4x4<float>  mf = SGLM::DemoMatrix<float>(2.f); 
    std::cout << "SGLM::DemoMatrix<double> md " << std::endl << SGLM::Present_<double>(md) << std::endl ;  
    std::cout << "SGLM::DemoMatrix<float>  mf"  << std::endl << SGLM::Present_<float>(mf) << std::endl ;  

    mf = md ; 
    std::cout << "SGLM::DemoMatrix<float>  mf (after mf = md )"  << std::endl << SGLM::Present_<float>(mf) << std::endl ;  
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_Narrow(); 

    return 0 ; 
}
