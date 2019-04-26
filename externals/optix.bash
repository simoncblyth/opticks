optix-source(){   echo ${BASH_SOURCE} ; }
optix-vi(){       vi $(optix-source) ; }
optix-env(){      olocal- ; }
optix-usage(){ cat << \EOU

NVIDIA OptiX Ray Trace Toolkit
================================== 

See Also
------------

* optixnote-  thousands of lines of lots of notes on OptiX versions and usage, that used to be here
    

Changing OptiX version
-------------------------

1. change envvar to point at desired install dir:: 

   OPTICKS_OPTIX_INSTALL_DIR=/usr/local/OptiX_600

2. do a clean build of okconf::

   cd ~/opticks/okconf
   om-clean 
   om-install

3. run the OKConfTest executable and check the expected versions appear

4. rebuild optickscore with changes to 

   om-clean optickscore
   om-install optickscore   
       ## link errors from OpticksBufferSpec are expected
       ## modify OpticksBufferSpec.hh for the new version



4. clean and install all subs from optixrap onwards::


   om-visit optixrap:      # just lists the subs, note the colon 
   om-clean optixrap:     
   om-install optixrap:    
   om-test optixrap:    


OptiX with multiple GPU
------------------------

CUDA_VISIBLE_DEVICES is honoured by OptiX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@localhost UseOptiX]$ CUDA_VISIBLE_DEVICES=0 UseOptiX
    OptiX 6.0.0
    Number of Devices = 1

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes

    [blyth@localhost UseOptiX]$ CUDA_VISIBLE_DEVICES=1 UseOptiX
    OptiX 6.0.0
    Number of Devices = 1

    Device 0: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes


    [blyth@localhost UseOptiX]$ CUDA_VISIBLE_DEVICES=0,1 UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes


nvidia-smi ignore CUDA_VISIBLE_DEVICES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


OptiX device ordinal not same as listed in nvidia-smi::

    blyth@localhost UseOptiX]$ nvidia-smi
    Wed Apr 17 15:43:12 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0  On |                  N/A |
    | 41%   32C    P8    18W / 280W |    225MiB / 24189MiB |      1%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:A6:00.0 Off |                  N/A |
    | 32%   47C    P8    28W / 250W |      0MiB / 12036MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0     13810      G   /usr/bin/X                                   149MiB |
    |    0     15683      G   /usr/bin/gnome-shell                          74MiB |
    +-----------------------------------------------------------------------------+



OptiX_600 optix-pdf : looking for new things
-----------------------------------------------


p9,10 : multi-GPU
~~~~~~~~~~~~~~~~~~~~

As of OptiX 4.0, mixed multi-GPU setups are available on all supported GPU architectures
which are Kepler, Maxwell, Pascal, and Volta GPUs.

By default all compatible GPU devices in a system will be selected in an OptiX context when
not explicitly using the function rtContextSetDevices to specify which devices should be
made available. If incompatible devices are selected an error is returned from
rtContextSetDevices.

In mixed GPU configurations, the kernel will be compiled for each streaming multiprocessor
(SM) architecture, extending the initial start-up time.

For best performance, use multi-GPU configurations consisting of the same GPU type. Also
prefer PCI-E slots in the system with the highest number of electrical PCI-E lanes (x16 Gen3
recommended).

On system configurations without NVLINK support, the board with the smallest VRAM
amount will be the limit for on-device resources in the OptiX context. In homogeneous
multi-GPU systems with NVLINK bridges and the driver running in the Tesla Compute
Cluster (TCC) mode, OptiX will automatically use peer-to-peer access across the NVLINK
connections to use the combined VRAM of the individual boards together which allows
bigger scene sizes.


p14 : Enabling RTX mode 
~~~~~~~~~~~~~~~~~~~~~~~~~

As of OptiX version 5.2, RTX mode can be enabled to take advantage of RT Cores,
accelerating ray tracing by computing traversal and triangle intersection in hardware.
RTX mode is not enabled by default. RTX mode can be enabled with the
RT_GLOBAL_ATTRIBUTE_ENABLE_RTX attribute using rtGlobalSetAttribute when creating the
OptiX context. However, certain features of OptiX will not be available.


:google:`RT_GLOBAL_ATTRIBUTE_ENABLE_RTX`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.ks.uiuc.edu/Research/vmd/doxygen/OptiXRenderer_8C-source.html
* https://raytracing-docs.nvidia.com/optix/api/html/group___context_free_functions.html


p27 : Selector nodes are deprecated in RTX mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: Selector nodes are deprecated in RTX mode. Future updates to RTX mode will
provide a mechanism to support most of the use cases that required Selector nodes. See
Enabling RTX mode (page 14).


p33 : RTgeometrytriangles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RTgeometrytriangles type provides OptiX with built-in support for triangles.
RTgeometrytriangles complements the RTgeometry type, with functions that can explicitly
define the triangle data. Custom intersection and bounding box programs are not required by
RTgeometrytriangles; the application only needs to provide the triangle data to OptiX.


p133 : Choose types that optimize writing to buffers.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In multi-GPU environments INPUT_OUTPUT and OUTPUT buffers are stored on the host. In
order to optimize writes to these buffers, types of either 4 bytes or 16 bytes (for example,
float, uint, or float4) should be used when possible. One might be tempted to make an
output buffer used for the screen float3 for an RGB image. However, using a float4
buffer instead will result in improved performance.




EOU
}

optix-export(){  echo -n ; }

optix-install-dir(){  echo $OPTICKS_OPTIX_INSTALL_DIR ; }
optix-name(){ echo $(basename $(optix-install-dir)) ; } 
optix-vers(){ local name=$(optix-name) ; echo ${name/OptiX_} ; }
optix-version(){ local vers=$(optix-vers) ; echo ${vers:0:1}.${vers:1:1}.${vers:2:1} ; }  ## assumes vers like 510 or 600 

optix-api-(){ echo $(optix-install-dir)/doc/OptiX_API_Reference_$(optix-version).pdf ; }
optix-pdf-(){ echo $(optix-install-dir)/doc/OptiX_Programming_Guide_$(optix-version).pdf ; }
optix-api(){ open $(optix-api-) ; }
optix-pdf(){ open $(optix-pdf-) ; }

optix-api-html-(){ echo https://raytracing-docs.nvidia.com/optix/api/html/index.html ; }
optix-api-html(){ open $(optix-api-html-) ; }






optix-dir(){          echo $(optix-install-dir) ; }
optix-idir(){         echo $(optix-install-dir)/include ; }

optix-c(){     cd $(optix-dir); }
optix-cd(){    cd $(optix-dir); }
optix-icd(){   cd $(optix-idir); }
optix-ifind(){ find $(optix-idir) -name '*.h' -exec grep ${2:--H} ${1:-setMiss} {} \; ; }

optix-info(){ cat << EOI

   optix-install-dir  : $(optix-install-dir)
   optix-dir          : $(optix-dir)
   optix-idir         : $(optix-idir)

   optix-name         : $(optix-name)
   optix-vers         : $(optix-vers)
   optix-version      : $(optix-version)

   optix-api-         : $(optix-api-)  
   optix-pdf-         : $(optix-pdf-)  

EOI
}

optix-cuda-nvcc-flags(){
    case $NODE_TAG in 
       D) echo -ccbin /usr/bin/clang --use_fast_math ;;
       *) echo --use_fast_math ;; 
    esac
}




optix-samples-notes(){ cat << EON
$FUNCNAME
======================

optix-samples-setup
     copy SDK directory of samples to SDK-src and make writable



use_tri_api
-----------

::

    [blyth@localhost SDK-src]$ find . -type f -exec grep -H use_tri_api {} \;
    ./optixMDLDisplacement/optixMDLDisplacement.cpp:    mesh.use_tri_api  = false;
    ./optixMotionBlur/optixMotionBlur.cpp:bool           use_tri_api = false;
    ./optixMotionBlur/optixMotionBlur.cpp:    mesh.use_tri_api = use_tri_api;
    ./optixMotionBlur/optixMotionBlur.cpp:        if( use_tri_api )
    ./optixMotionBlur/optixMotionBlur.cpp:            use_tri_api = true;
    ./sutil/OptiXMesh.h:    : use_tri_api( true )
    ./sutil/OptiXMesh.h:  bool                         use_tri_api;   // optional
    ./sutil/OptiXMesh.cpp:  if( optix_mesh.use_tri_api )
    ./optixMeshViewer/optixMeshViewer.cpp:bool           use_tri_api = true;
    ./optixMeshViewer/optixMeshViewer.cpp:    mesh.use_tri_api = use_tri_api;
    ./optixMeshViewer/optixMeshViewer.cpp:            use_tri_api = false;
    [blyth@localhost SDK-src]$ 


::

    239   if( optix_mesh.use_tri_api )
    240   {
    241     optix::GeometryTriangles geom_tri = ctx->createGeometryTriangles();
    242     geom_tri->setPrimitiveCount( mesh.num_triangles );
    243     geom_tri->setTriangleIndices( buffers.tri_indices, RT_FORMAT_UNSIGNED_INT3 );
    24



EON
}

optix-sfind(){    optix-samples-scd ; find . \( -name '*.cu' -or -name '*.h' -or -name '*.cpp'  \) -exec grep ${2:--H} "${1:-rtReport}" {} \; ; }

optix-scd(){ optix-samples-scd  ; }

optix-samples-sdir(){ echo $(optix-install-dir)/SDK-src ; }
optix-samples-bdir(){ echo $(optix-install-dir)/SDK-src.build ; }
optix-samples-scd(){ cd $(optix-samples-sdir) ; }
optix-samples-bcd(){ cd $(optix-samples-bdir) ; }
optix-samples-setup(){

   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local idir=$(optix-install-dir)
   local sdir=$(optix-samples-sdir)

   [ -d "$sdir" ] && echo $msg already setup in $sdir && return

   cd $idir
   local cmd="sudo cp -R SDK SDK-src && sudo chown $USER SDK-src && sudo mkdir SDK-src.build && sudo chown $USER SDK-src.build "
   echo $cmd
   eval $cmd

   cd $iwd
}

optix-samples-cmake(){
    local iwd=$PWD
    local bdir=$(optix-samples-bdir)
    #rm -rf $bdir   # starting clean 
    mkdir -p $bdir
    cd $bdir

    cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) \
          -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
           "$(optix-samples-sdir)"

    cd $iwd
}

optix-samples-make()
{
    local iwd=$PWD
    optix-samples-bcd
    make -j$(nproc)
    cd $iwd
}

optix-samples-run(){
    local name=${1:-materials}
    optix-samples-make $name
    local cmd="$(optix-bdir)/bin/$name"
    echo $cmd
    eval $cmd
}


optix-samples-cmake-notes(){ cat << EON


-- Found CUDA: /usr/local/cuda-10.1 (found suitable version "10.1", minimum required is "5.0") 
CMake Warning (dev) at /usr/share/cmake3/Modules/FindOpenGL.cmake:270 (message):
  Policy CMP0072 is not set: FindOpenGL prefers GLVND by default when
  available.  Run "cmake --help-policy CMP0072" for policy details.  Use the
  cmake_policy command to set the policy and suppress this warning.

  FindOpenGL found both a legacy GL library:

    OPENGL_gl_LIBRARY: /usr/lib64/libGL.so

  and GLVND libraries for OpenGL and GLX:

    OPENGL_opengl_LIBRARY: /usr/lib64/libOpenGL.so
    OPENGL_glx_LIBRARY: /usr/lib64/libGLX.so

  OpenGL_GL_PREFERENCE has not been set to "GLVND" or "LEGACY", so for
  compatibility with CMake 3.10 and below the legacy GL library will be used.
Call Stack (most recent call first):
  CMake/FindSUtilGLUT.cmake:35 (find_package)
  CMakeLists.txt:261 (include)
This warning is for project developers.  Use -Wno-dev to suppress it.


EON
}





optix600-install-experimental()
{
    ## for packaging purposes need to try treating OptiX more like any other external
    cd /usr/local
    local prefix=$LOCAL_BASE/opticks/externals/optix
    mkdir -p $prefix
    echo need to say yes then no to the installer
    sh NVIDIA-OptiX-SDK-6.0.0-linux64-25650775.sh --prefix=$prefix
}



