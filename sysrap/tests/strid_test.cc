// ./strid_test.sh 

#include <cassert>
#include <iostream>
#include <glm/glm.hpp>
#include "strid.h"

void set( glm::tvec4<uint64_t>& col3, int mode)
{
    if( mode == 0 )
    {
        col3.x = 0 ; 
        col3.y = 0 ; 
        col3.z = 0 ; 
        col3.w = 0 ; 
    }
    else if(mode == 1)
    {
        col3.x = 0xffffffffff0fffff ; 
        col3.y = 0xaaaaaaaafff0ffff ; 
        col3.z = 0xbbbbbbbbffff0fff ; 
        col3.w = 0xccccccccfffff0ff ; 
    }
    else if(mode == -1)
    {
        col3.x = -1 ; 
        col3.y = 0xffffffffffffffff ; 
        col3.z = -1 ; 
        col3.w = 0xffffffffffffffff ; 
    }
}

void test_Encode_Decode(int mode=0)
{
    std::cout << "test_Encode_Decode " << mode << std::endl ;

    glm::tmat4x4<double> tr(1.); 
    assert( strid::IsClear(tr)==true );

    glm::tvec4<uint64_t> col3 ;  
    set(col3, mode); 
     
    strid::Encode(tr, col3 ); 
    assert( strid::IsClear(tr)==false );
    
    glm::tvec4<uint64_t> col3_ ; 
    strid::Decode(tr, col3_ ); 

    for(unsigned r=0 ; r < 4 ; r++) assert( col3[r] == col3_[r] ); 

    std::cout << strid::Desc<double, uint64_t>(tr) << std::endl ; 
}

void test_Narrow(int mode)
{
    std::cout << "test_Narrow " << mode << std::endl ;

    glm::tmat4x4<double> src(1.); 

    glm::tvec4<uint64_t> col3 ; 
    set(col3, mode); 

    strid::Encode(src, col3 ); 
    std::cout << "src\n" << strid::Desc<double, uint64_t>(src) << std::endl ; 

    glm::tmat4x4<float>  dst(1.); 
    strid::Narrow(dst, src); 
    std::cout << "dst\n" << strid::Desc<float,  uint32_t>(dst) << std::endl ; 
}



int main(int argc, char** argv)
{
    for(int mode=-1 ; mode < 2 ; mode++)
    {
        test_Encode_Decode(mode); 
        test_Narrow(mode);  
    }

    return 0 ; 
}
