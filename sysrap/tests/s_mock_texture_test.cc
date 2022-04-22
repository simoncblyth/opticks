// name=s_mock_texture_test  ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include  -o /tmp/$name && /tmp/$name

#include <cstdio>
#include "s_mock_texture.h"

int main(int argc, char** argv)
{
    MockTextureManager mgr ; 

    cudaTextureObject_t tex = 1 ;  

    float x = 0.5f ; 
    float y = 0.5f ; 
    float v = tex2D<float>(tex, x, y ) ; 
    printf("// v %10.4f \n", v ); 

    float4 v4 = tex2D<float4>(tex, x, y ) ; 
    printf("// v4 (%10.4f %10.4f %10.4f %10.4f)  \n", v4.x, v4.y, v4.z, v4.w  ); 

    return 0;
}

