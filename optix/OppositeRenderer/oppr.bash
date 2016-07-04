# === func-gen- : cuda/optix/OppositeRenderer/oppr fgp cuda/optix/OppositeRenderer/oppr.bash fgn oppr fgh cuda/optix/OppositeRenderer
oppr-src(){      echo optix/OppositeRenderer/oppr.bash ; }
oppr-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oppr-src)} ; }
oppr-vi(){       vi $(oppr-source) ; }
oppr-env(){      elocal- ; }
oppr-usage(){ cat << EOU

* http://apartridge.github.io/OppositeRenderer/

Example of a photon mapping project using OptiX, Thrust, ...

::

    simon:~ blyth$ oppr-
    simon:~ blyth$ oppr-cd
    simon:OppositeRenderer blyth$ find . -type f -exec grep -H thrust {} \;
    ./OppositeRenderer/RenderEngine/RenderEngine.vcxproj.filters:    <Filter Include="sutil\thrust">
    ./OppositeRenderer/RenderEngine/renderer/helpers/optix.h:static thrust::device_ptr<T> getThrustDevicePtr(optix::Buffer & buffer, int deviceNumber)
    ./OppositeRenderer/RenderEngine/renderer/helpers/optix.h:    return thrust::device_pointer_cast(getDevicePtr<T>(buffer, deviceNumber));
    ./OppositeRenderer/RenderEngine/renderer/OptixRenderer_SpatialHash.cu:#include <thrust/reduce.h>
    ./OppositeRenderer/RenderEngine/renderer/OptixRenderer_SpatialHash.cu:#include <thrust/pair.h>
    ./OppositeRenderer/RenderEngine/renderer/OptixRenderer_SpatialHash.cu:#include <thrust/device_vector.h>
    ./OppositeRenderer/RenderEngine/renderer/OptixRenderer_SpatialHash.cu:#include <thrust/host_vector.h>
    ...  



RenderEngine/renderer/helpers/optix.h::

    #pragma  once

    template<typename T>
    static T* getDevicePtr(optix::Buffer & buffer, int deviceNumber)
    {
        CUdeviceptr d;
        buffer->getDevicePointer(deviceNumber, &d);
        return (T*)d;
    }

    template<typename T>
    static thrust::device_ptr<T> getThrustDevicePtr(optix::Buffer & buffer, int deviceNumber)
    {
        return thrust::device_pointer_cast(getDevicePtr<T>(buffer, deviceNumber));
    }


/usr/local/env/cuda/optix/OppositeRenderer/OppositeRenderer/RenderEngine/renderer/


* OptixRenderer.h 
* OptixRenderer.cpp
* OptixRenderer_SpatialHash.cu

Interesting structure... methods of OptixRenderer are mixed up OptixRenderer_SpatialHash.cu with 
thrust structs, CUDA kernels  etc..  but there is no header

Global functions like createUniformGridPhotonMap(PPMRadius);without any header, 

Methods from are invoked from       



Photon maps used for global illumination can get away with being rather coarse, as
most of the scene detail comes from direct ray tracing and the global illumination just
fills out shadows, caustics

OptixRenderer.cpp::

     30 #if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID
     31 const unsigned int OptixRenderer::PHOTON_GRID_MAX_SIZE = 100*100*100;   


For photon simulation the photon map is much more important than the geometry.





EOU
}
oppr-dir(){ echo $(local-base)/env/cuda/optix/OppositeRenderer ; }
oppr-cd(){  cd $(oppr-dir); }
oppr-mate(){ mate $(oppr-dir) ; }
oppr-get(){
   local dir=$(dirname $(oppr-dir)) &&  mkdir -p $dir && cd $dir
   git clone https://github.com/apartridge/OppositeRenderer.git 
}

oppr-scene(){
   vi $(oppr-dir)/OppositeRenderer/RenderEngine/scene/Scene.{h,cpp}
}
oppr-renderer(){
   vi $(oppr-dir)/OppositeRenderer/RenderEngine/renderer/OptixRenderer.{h,cpp}
}


