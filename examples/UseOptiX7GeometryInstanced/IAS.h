#pragma once

#include <optix.h>
#include <vector>
#include <glm/glm.hpp>

/**
IAS
====

TODO: 

* generalize to multiple GAS handles referenced from a single IAS : 
  perhaps with similar API to lighthouse2, see::
 
     env-;lighthouse2-;lighthouse2-vi 

* aim for maximally flat structure in order  
  for the traversal to be handled by RT cores  

* replace OptixTraversableHandle with an index into an array 

**/


struct IAS
{
    OptixTraversableHandle gas_handle ;   // geometry to be repeated
    OptixTraversableHandle ias_handle ; 
    CUdeviceptr            d_as_output_buffer;

    std::vector<OptixInstance> instances ; 

    OptixTraversableHandle handle;
    CUdeviceptr            d_instances ;

    void addInstance(const glm::mat4& mat);
    void dump(const float* imp); 

    IAS(OptixTraversableHandle gas_handle_);
 
    void init(); 
    void initInstancesOne();
    void initInstancesTwo();
    void initInstancesMany();

    void build(); 
    OptixTraversableHandle build(OptixBuildInput buildInput); 

};


