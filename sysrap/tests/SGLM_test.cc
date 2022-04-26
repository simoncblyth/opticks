// name=SGLM_test ; glm- ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I$(glm-idir) -o /tmp/$name && /tmp/$name

#include <cassert>
#include "SGLM.hh"

void test_GetEVec()
{
    glm::vec3 v(0.f, 0.f, 0.f ); 
    SGLM::GetEVec(v, "VEC3" , "1,2,3" );  

    assert( v.x == 1.f ); 
    assert( v.y == 2.f ); 
    assert( v.z == 3.f ); 

    const char* key = "VEC4" ; 

    glm::vec4 f(SGLM::EVec4(key, "10,20,30,40"));

    std::cout << std::setw(10) << key << SGLM::Present(f) << std::endl ; 
   
    assert( f.x == 10.f ); 
    assert( f.y == 20.f ); 
    assert( f.z == 30.f ); 
    assert( f.w == 40.f ); 
}


void test_GetMVP()
{
    glm::mat4 world2clip = SGLM::GetMVP( 1024, 768, true );  
    std::cout << "world2clip\n" << SGLM::Present(world2clip) << std::endl ; 

}

void test_SGLM_cf()
{
    SGLM sglm ; 
    sglm.dump(); 
    
    glm::mat4 GetMVP_world2clip = SGLM::GetMVP( 1024, 768, false );  
    std::cout << "GetMVP_world2clip\n" << SGLM::Present(GetMVP_world2clip) << std::endl ; 
    std::cout << "sglm.world2clip\n" << SGLM::Present(sglm.world2clip) << std::endl ; 
}

void test_SGLM_update()
{
    SGLM sglm ; 
    sglm.dump(); 

    sglm.zoom = 10.f ; 
    sglm.update(); 
    sglm.dump(); 

}



int main(int argc, char** argv)
{
    //test_GetEVec(); 
    //test_GetMVP(); 
    //test_SGLM_cf();
 
    test_SGLM_update(); 
 
    return 0 ; 
}
