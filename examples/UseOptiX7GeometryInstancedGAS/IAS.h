#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <optix.h>

struct IAS
{
    CUdeviceptr             d_buffer;    
    CUdeviceptr             d_instances ;   
    OptixTraversableHandle  handle ; 

    static IAS Build(const std::vector<float>& tr ); // must contain a multiple of 16 transform floats with integer identity in spare slots 
    static IAS Build(const std::vector<glm::mat4>& tr ); // must contain a multiple of 16 transform floats with integer identity in spare slots 
    static IAS Build(const float* vals, unsigned num_vals); 
};


