# === func-gen- : numerics/thrustrap/thrustrap fgp numerics/thrustrap/thrustrap.bash fgn thrustrap fgh numerics/thrustrap
thrustrap-src(){      echo numerics/thrustrap/thrustrap.bash ; }
thrustrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(thrustrap-src)} ; }
thrustrap-vi(){       vi $(thrustrap-source) ; }
thrustrap-env(){      elocal- ; }
thrustrap-usage(){ cat << EOU


Observations from the below match my experience

* https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks

CUDA 5.5 has problems with c++11 ie libc++ on Mavericks, 
can only get to compile and run without segv only by 

* targetting the older libstdc++::

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -stdlib=libstdc++")

* not linking to other libs built against c++11 ie libc++

delta:tests blyth$ otool -L /usr/local/env/numerics/thrustrap/bin/ThrustEngineTest
/usr/local/env/numerics/thrustrap/bin/ThrustEngineTest:
    @rpath/libcudart.5.5.dylib (compatibility version 0.0.0, current version 5.5.28)
    /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 60.0.0)
    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)



CUDA 7 to the rescue ?
------------------------

* http://devblogs.nvidia.com/parallelforall/cuda-7-release-candidate-feature-overview/



Can a C interface provide a firewall to allow interop between compilers ?
===========================================================================


CUDA OpenGL thrust interop
----------------------------

* https://gist.github.com/dangets/2926425

::

    #include <cuda_gl_interop.h>
    #include <thrust/device_vector.h>


    // initialization

    GLuint vbo;
    struct cudaGraphicsResource *vbo_cuda;

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo, cudaGraphicsMapFlagsWriteDiscard);


    // display time : handover OpenGL -> CUDA/thrust 

    cudaGraphicsMapResources(/*count*/1, &vbo_cuda,/*stream*/ 0);

    float4 *raw_ptr;
    size_t buf_size;
    cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, vbo_cuda);

    thrust::device_ptr<float4> dev_ptr = thrust::device_pointer_cast(raw_ptr);
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last(g_mesh_width * g_mesh_height);
    thrust::transform(first, last, dev_ptr, sine_wave(g_mesh_width, g_mesh_height, g_anim));
 
    cudaGraphicsUnmapResources(1, &vbo_cuda, 0);  // CUDA/thrust back -> OpenGL
 


OptiX / OpenGL interop
------------------------


::

    OptiXEngine::init creates OptiX buffers using OpenGL buffer_id 

    m_genstep_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, genstep_buffer_id);


    GMergedMeshOptiXGeometry uses OptiX buffer map/unmap when copying data into buffers


/Developer/OptiX/include/optixu/optixpp_namespace.h::

    1485     /// Get the pointer to buffer memory on a specific device. See @ref rtBufferGetDevicePointer
    1486     void getDevicePointer( unsigned int optix_device_number, CUdeviceptr *device_pointer );
    1487     CUdeviceptr getDevicePointer( unsigned int optix_device_number );



* https://devtalk.nvidia.com/default/topic/551556/?comment=3858139


Histogramming Check
----------------------

::

    In [1]: h = phc_(1)
    INFO:env.g4dae.types:loading /usr/local/env/phcerenkov/1.npy 
    -rw-r--r--  1 blyth  staff  4902808 Jun 27 18:30 /usr/local/env/phcerenkov/1.npy

    In [2]: h[:,0,0]
    Out[2]: array([ 3265, 15297,     5, ...,     3,     3,     3], dtype=uint64)

    In [3]: hh = h[:,0,0]

   In [18]: uhh = np.unique(hh)

    In [19]: map(hex_, uhh)  # huh ? where the fffff from 
    Out[19]: 
    ['0x3',
     '0x5',
     '0x31',
     '0x51',
     '0x61',
     '0xc1',
     '0xf1',
     '0x361',
     '0x3b1',
     '0x3c1',
     '0x551',
     '0x561',
    ...
     '0x6cccc551',
     '0x6cccc561',
     '0x6cccc5c1',
     '0x6cccc651',
     '0x6ccccc51',
     '0x6cccccc1',
     '0xffffffffb55cc551',
     '0xffffffffb56ccc51',
     '0xffffffffb5b5c551',
     '0xffffffffb5bb5c51',
     '0xffffffffb5cc5c51',
     '0xffffffffb5cccc51',



      TODO:Compare with thrust...

      /usr/local/env/numerics/thrustrap/bin/PhotonIndexTest
    


Looks like no mapping/unmapping needed so long as dont change the size of the buffer


OpenGL buffer objects like PBOs and VBOs can be encapsulated for use in OptiX
with rtBufferCreateFromGLBO. The resulting buffer is a reference only to the
OpenGL data; the size of the OptiX buffer as well as the format have to be set
via rtBufferSetSize and rtBufferSetFormat. When the OptiX buffer is destroyed,
the state of the OpenGL buffer object is unaltered. Once an OptiX buffer is
created, the original GL buffer object is immutable, meaning the properties of
the GL object like its size cannot be changed while registered with OptiX.
However, it is still possible to read and write to the GL buffer object using
the appropriate GL functions. If it is necessary to change properties of an
object, first call rtBufferGLUnregister before making changes. After the
changes are made the object has to be registered again with rtBufferGLRegister.
This is necessary to allow OptiX to access the objects data again. Registration
and unregistration calls are expensive and should be avoided if possible.






EOU
}

thrustrap-sdir(){ echo $(env-home)/numerics/thrustrap ; }
thrustrap-idir(){ echo $(local-base)/env/numerics/thrustrap ; }
thrustrap-bdir(){ echo $(thrustrap-idir).build ; }

thrustrap-scd(){  cd $(thrustrap-sdir); }
thrustrap-cd(){   cd $(thrustrap-sdir); }

thrustrap-icd(){  cd $(thrustrap-idir); }
thrustrap-bcd(){  cd $(thrustrap-bdir); }
thrustrap-name(){ echo ThrustRap ; }

thrustrap-wipe(){
   local bdir=$(thrustrap-bdir)
   rm -rf $bdir
}

thrustrap-env(){  
   elocal- 
   cuda-
   cuda-export
   #optix-
   #optix-export
   thrust-
   thrust-export 
}

thrustrap-cmake(){
   local iwd=$PWD
   local msg="=== $FUNCNAME : "
   local bdir=$(thrustrap-bdir)
   mkdir -p $bdir
  
   thrustrap-bcd 

   local flags=$(cuda-nvcc-flags)
   echo $msg using CUDA_NVCC_FLAGS $flags

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(thrustrap-idir) \
       -DCUDA_NVCC_FLAGS="$flags" \
       $(thrustrap-sdir)

   cd $iwd
}

thrustrap-make(){
   local iwd=$PWD

   thrustrap-bcd
   make $*

   cd $iwd
}

thrustrap-install(){
   thrustrap-make install
}

thrustrap-bin(){ echo $(thrustrap-idir)/bin/$(thrustrap-name)Test ; }
thrustrap-export()
{ 
   echo -n 
}
thrustrap-run(){
   local bin=$(thrustrap-bin)
   thrustrap-export
   $bin $*
}



thrustrap--()
{
    thrustrap-wipe
    thrustrap-cmake
    thrustrap-make
    thrustrap-install

}



