// ./strid_test.sh 

#include <cassert>
#include <iostream>
#include <glm/glm.hpp>
#include "strid.h"

void test_Encode_Decode_4()
{
    glm::tmat4x4<double> tr(1.); 
    assert( strid::IsClear(tr)==true );

    uint64_t e03 = 0xffffffffff0fffff ; 
    uint64_t e13 = 0xaaaaaaaafff0ffff ; 
    uint64_t e23 = 0xbbbbbbbbffff0fff ; 
    uint64_t e33 = 0xccccccccfffff0ff ; 

    strid::Encode(tr, e03, e13, e23, e33 ); 
    assert( strid::IsClear(tr)==false );
    
    uint64_t e03_, e13_, e23_, e33_ ; 
    strid::Decode(tr, e03_, e13_, e23_, e33_ ); 

    assert( e03 == e03_ ); 
    assert( e13 == e13_ ); 
    assert( e23 == e23_ ); 
    assert( e33 == e33_ ); 

    std::cout << strid::Desc(tr) << std::endl ; 
}


void test_Encode_Decode_1()
{
    glm::tmat4x4<double> tr(1.); 
    assert( strid::IsClear(tr)==true );

    uint64_t e03 = 0xffffffffff0fffff ; 
    strid::Encode(tr, e03 ); 
    uint64_t e03_ ; 
    strid::Decode(tr, e03_ ); 
    assert( e03 == e03_ ); 
 
    std::cout << strid::Desc(tr) << std::endl ; 
}




int main(int argc, char** argv)
{
    //test_Encode_Decode_4(); 
    test_Encode_Decode_1(); 

    return 0 ; 
}
