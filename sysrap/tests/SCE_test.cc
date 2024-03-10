/**
SCE_test.cc
============

::

    ~/o/sysrap/tests/SCE_test.sh 

**/

#include "SCE.h"
#include "SPresent.h"
#include "scuda.h"

int main()
{
    glm::tvec4<float> ce(0., 0., 0., 100. ); 
    float4 _ce = make_float4( ce.x, ce.y, ce.z, ce.w ); 

    std::vector<glm::tvec4<float>> corners ; 
    SCE::Corners(corners, ce ); 
    std::cout << "corners" << std::endl << SPresent(corners) << std::endl ; 

    std::vector<float4> _corners ; 
    SCE::Corners(_corners, _ce ); 
    std::cout << "_corners" << std::endl << SPresent(_corners) << std::endl ; 

    std::vector<float4> __corners(corners.size()) ;
    assert( sizeof(glm::tvec4<float>) == sizeof(float4)); 
    memcpy(__corners.data(), corners.data(), corners.size()*sizeof(float4) ); 
    std::cout << "__corners" << std::endl << SPresent(__corners) << std::endl ; 



    std::vector<glm::tvec4<float>> midface ; 
    SCE::Midface(midface, ce ); 
    std::cout << "midface" << std::endl << SPresent(midface) << std::endl ; 

    std::vector<float4> _midface ; 
    SCE::Midface(_midface, _ce ); 
    std::cout << "_midface" << std::endl << SPresent(_midface) << std::endl ; 


    return 0 ; 
}
