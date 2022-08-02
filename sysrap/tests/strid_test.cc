// ./strid_test.sh 

#include <cassert>
#include <iostream>
#include <glm/glm.hpp>
#include "strid.h"

void test_Encode_Decode()
{
    glm::tmat4x4<double> tr(1.); 
    assert( strid::IsClear(tr)==true );

    glm::tvec4<uint64_t> col3 ; 
    col3.x = 0xffffffffff0fffff ; 
    col3.y = 0xaaaaaaaafff0ffff ; 
    col3.z = 0xbbbbbbbbffff0fff ; 
    col3.w = 0xccccccccfffff0ff ; 
  
    strid::Encode(tr, col3 ); 
    assert( strid::IsClear(tr)==false );
    
    glm::tvec4<uint64_t> col3_ ; 
    strid::Decode(tr, col3_ ); 

    for(unsigned r=0 ; r < 4 ; r++) assert( col3[r] == col3_[r] ); 

    std::cout << strid::Desc<double, uint64_t>(tr) << std::endl ; 
}

void test_Narrow()
{
    glm::tmat4x4<double> src(1.); 

    glm::tvec4<uint64_t> col3 ; 
    col3.x = 0xffffffffff0fffff ; 
    col3.y = 0xaaaaaaaafff0ffff ; 
    col3.z = 0xbbbbbbbbffff0fff ; 
    col3.w = 0xccccccccfffff0ff ; 
    strid::Encode(src, col3 ); 
    std::cout << "src\n" << strid::Desc<double, uint64_t>(src) << std::endl ; 
 

    glm::tmat4x4<float>  dst(1.); 
    strid::Narrow(dst, src); 
    std::cout << "dst\n" << strid::Desc<float,  uint32_t>(dst) << std::endl ; 

}



int main(int argc, char** argv)
{
    test_Encode_Decode(); 
    /*
    */

    test_Narrow();  


    return 0 ; 
}
