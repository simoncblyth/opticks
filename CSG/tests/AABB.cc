
// ./qat4.sh 

#include "sutil_vec_math.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#include "AABB.h"
#include "qat4.h"


void test_cube_corners()
{
    float4 ce = make_float4( 0.f , 0.f, 0.f, 1.f ); 
    std::vector<float3> corners ; 
    AABB::cube_corners(corners, ce); 
    for(int i=0 ; i < int(corners.size()) ; i++) std::cout << i << ":" << corners[i] << std::endl ; 

    qat4 q ; 
    q.q3.f.x = 100.f ; 
    q.q3.f.y = 100.f ; 
    q.q3.f.z = 100.f ; 

    AABB bb = {} ; 

    q.right_multiply_inplace( corners, 1.f ); 
    for(int i=0 ; i < int(corners.size()) ; i++) std::cout << i << ":" << corners[i] << std::endl ; 
    for(int i=0 ; i < int(corners.size()) ; i++) bb.include_point(corners[i]) ; 
    std::cout << " bb " << bb.desc() << std::endl ; 
}


int main(int argc, char** argv)
{
    test_cube_corners(); 
    return 0 ; 
}


