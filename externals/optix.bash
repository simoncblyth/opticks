optix-source(){   echo ${BASH_SOURCE} ; }
optix-vi(){       vi $(optix-source) ; }
optix-env(){      olocal- ; }
optix-usage(){ cat << \EOU

NVIDIA OptiX Ray Trace Toolkit
================================== 

See Also
------------

* optixnote-  thousands of lines of lots of notes on OptiX versions and usage, that used to be here


* https://devtalk.nvidia.com/default/board/254/optix/
* http://raytracing-docs.nvidia.com/optix/index.html
* https://raytracing-docs.nvidia.com/optix_6_0/index.html
* https://raytracing-docs.nvidia.com/optix_6_0/tutorials_6_0/index.html#preface#

    

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



Stack Size
---------------

* https://devtalk.nvidia.com/default/topic/1004649/optix/code-work-well-in-optix-3-9-1-but-fail-in-optix-4-0-2/post/5130084/#5130084


 No, it's quite the opposite. Your stack size is much too big and you're running out of VRAM on the board just for that.

To find out which stack size is enough do the following:
1.) Add an exception program which prints the exception code and enable it for the whole launch size.
You'll find links to my code doing that when searching this forum for "rtAssert", and here: https://devtalk.nvidia.com/default/topic/936762/?comment=4882450
2.) Maybe add a command line variable to your application to set the OptiX stack size. That will make the following iterations quicker.
3.) Shrink the stack size to a small value, like 1024 or even less.
4.) Run the program in a configuration which uses the maximum intended number of recursions and check for stack overflow exceptions. (Start with a small stack size which throws some.)
5.) Increase the stack size until the exceptions in step 4. do not happen anymore.

Use the minimal stack size which doesn't show stack overflow exceptions anymore and still runs the program.

When reporting OptiX issues please always list the following system information:
OS version, installed GPU(s) (=> how much VRAM?), display driver version, OptiX version, CUDA toolkit version.

#2
Posted 04/18/2017 08:04 AM   


Exceptions
----------

* https://devtalk.nvidia.com/default/topic/936762/optix/distance-field-camera-clipping-issue-julia-demo-/post/4882450/#4882450


    #if USE_DEBUG_EXCEPTIONS
        // Disable this by default for performance, otherwise the stitched PTX code will have lots of exception handling inside. 
        m_context->setPrintEnabled(true);
        m_context->setPrintLaunchIndex(256, 256); // Launch index (0,0) at lower left.
        m_context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
    #endif

    rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

    RT_PROGRAM void exception()
    {
    #if USE_DEBUG_EXCEPTIONS
      const unsigned int code = rtGetExceptionCode();
      rtPrintf("Exception code 0x%X at (%d, %d)\n", code, launchIndex.x, launchIndex.y);
    #endif
    }






nvrtc : runtime compilation for OptiX 
---------------------------------------------------------------------------------------------------------------

* thinking about geometry time dynamic code generation of SDF functions, for use 
  in raymarching/sphere tracing 

* OptiX 6.0.0 SDK samples use NVRTC

30min in see options needed for runtime compilation with nvrtc from Detlef::

   http://on-demand.gputechconf.com/gtc/2017/video/s7185-mank-roettger-leveraging-nvrtc-runtime-compliation-for-dynamically-building-optix-shaders-from-mdl.mp4


Dome Camera (180 degree)
-------------------------

* http://on-demand.gputechconf.com/gtc/2015/presentation/S5246-David-McAllister.pdf

::

    float2 d = make_float2(launch_index)/make_float2(screen)*make_float2(2.0f,2.0f)-make_float2(1.0f, 1.0f); 
    float3 angle = make_float3 ( d.x , d.y , sqrtf(1.0f - ( d.x*d.x + d.y*d.y))); 
    float3 ray_direction = normalize( angle.x*normalize(U) + angle.y*normalize(V) + angle.z*normalize(W)); 
    optix::Ray ray(ray_origin,ray_direction,radiance_ray_type,scene_epsilon); 


VisRTX : C++ rendering framework developed by the HPC Visualization Developer Technology team at NVIDIA
----------------------------------------------------------------------------------------------------------

* https://github.com/NVIDIA/VisRTX
* https://gitmemory.com/tbiedert
* https://hpcvis.org/

* https://developer.nvidia.com/mdl-sdk

The MDL wrapper provided by VisRTX is self-contained and can be of interest to
anyone who wants to access the MDL SDK from an OptiX-based application.


Found this project by 

* :google:`optix DISABLE_ANYHIT`

See env- visrtx-


disabling ANYHIT : seems OptiX 6 has gained lots of flags and visibilityMasks 
------------------------------------------------------------------------------------

David::

    > => In OptiX 6, for best performance, you can actively disable anyhit, using one 
    > of the *_DISABLE_ANYHIT instance or ray flags. (Meaning, you can disable anyhit
    > on geometry, or alternatively, you can disable anyhit during trace.) If you 
    > aren't using an anyhit program, please try disabling anyhit and see if that
    > helps.


* https://github.com/NVIDIA/VisRTX/blob/10bbc184fa4d6dfc40154901ccb12461d779ca2d/src/Pathtracer/Pathtracer.cu

::

    #if OPTIX_VERSION_MAJOR >= 6
    const RTrayflags rayFlags = (launchParameters[0].disableAnyHit > 0) ? RT_RAY_FLAG_DISABLE_ANYHIT : RT_RAY_FLAG_NONE;
    rtTrace(/*launchParameters[0].*/topObject, ray, prd, RT_VISIBILITY_ALL, rayFlags);
    #else
    rtTrace(/*launchParameters[0].*/topObject, ray, prd);
    #endi


8.12.4.14
optix_declarations.h File Reference
enum RTgeometryflags
Material-dependent flags set on Geometry/GeometryTriangles.
Enumerator
RT_GEOMETRY_FLAG_NONE No special flags set.
RT_GEOMETRY_FLAG_DISABLE_ANYHIT Opaque flag, any hit program will be skipped.
RT_GEOMETRY_FLAG_NO_SPLITTING Disable primitive splitting to avoid potential duplicate
any hit program execution for a single intersection.


enum RTinstanceflags
Instance flags which override the behavior of geometry.

RT_INSTANCE_FLAG_DISABLE_ANYHIT Disable any-hit programs. This may yield
significantly higher performance even in cases where no any-hit programs are set.


8.12.4.23
enum RTrayflags
Ray flags.
Enumerator
RT_RAY_FLAG_NONE
RT_RAY_FLAG_DISABLE_ANYHIT Disables any-hit programs for the ray.
RT_RAY_FLAG_ENFORCE_ANYHIT Forces any-hit program execution for the ray.
RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT Terminates the ray after the first hit.
RT_RAY_FLAG_DISABLE_CLOSESTHIT Disables closest-hit programs for the ray.
RT_RAY_FLAG_CULL_BACK_FACING_TRIANGLES Do not intersect triangle back faces.
RT_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES Do not intersect triangle front faces.
RT_RAY_FLAG_CULL_DISABLED_ANYHIT Do not intersect geometry which disables any-hit
programs.
RT_RAY_FLAG_CULL_ENFORCED_ANYHIT Do not intersect geometry which enforces
any-hit programs.







::

    [blyth@localhost include]$ optix-ifind ANYHIT
    /home/blyth/local/opticks/externals/OptiX/include/internal/optix_declarations.h:  RT_GEOMETRY_FLAG_DISABLE_ANYHIT  = 0x01, /*!< Opaque flag, any hit program will be skipped */
    /home/blyth/local/opticks/externals/OptiX/include/internal/optix_declarations.h:  RT_INSTANCE_FLAG_DISABLE_ANYHIT           = 1u << 2,  /*!< Disable any-hit programs.
    /home/blyth/local/opticks/externals/OptiX/include/internal/optix_declarations.h:  RT_INSTANCE_FLAG_ENFORCE_ANYHIT           = 1u << 3   /*!< Override @ref RT_GEOMETRY_FLAG_DISABLE_ANYHIT */
    /home/blyth/local/opticks/externals/OptiX/include/internal/optix_declarations.h:  RT_RAY_FLAG_DISABLE_ANYHIT                = 1u << 0, /*!< Disables any-hit programs for the ray. */
    /home/blyth/local/opticks/externals/OptiX/include/internal/optix_declarations.h:  RT_RAY_FLAG_ENFORCE_ANYHIT                = 1u << 1, /*!< Forces any-hit program execution for the ray. */
    /home/blyth/local/opticks/externals/OptiX/include/internal/optix_declarations.h:  RT_RAY_FLAG_CULL_DISABLED_ANYHIT          = 1u << 6, /*!< Do not intersect geometry which disables any-hit programs. */
    /home/blyth/local/opticks/externals/OptiX/include/internal/optix_declarations.h:  RT_RAY_FLAG_CULL_ENFORCED_ANYHIT          = 1u << 7  /*!< Do not intersect geometry which enforces any-hit programs. */
    /home/blyth/local/opticks/externals/OptiX/include/optix_host.h:  * Setting the flags RT_GEOMETRY_FLAG_NO_SPLITTING and/or RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be dependent on the 
    /home/blyth/local/opticks/externals/OptiX/include/optix_host.h:  * RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be set for material index 0, if M0 and M2 allow it. 
    /home/blyth/local/opticks/externals/OptiX/include/optix_host.h:  * RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be set for material index 1, if M1 and M3 allow it. 
    /home/blyth/local/opticks/externals/OptiX/include/optix_host.h:  * RT_GEOMETRY_FLAG_DISABLE_ANYHIT is an optimization due to which the execution of the any hit program is skipped.
    [blyth@localhost include]$ 









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

#optix-install-dir(){  echo $OPTICKS_OPTIX_INSTALL_DIR ; }
optix-install-dir(){  echo $(opticks-prefix)/externals/OptiX  ; }
optix-name(){ echo $(readlink $(optix-install-dir)) ; } 
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
optix-samples-pdir(){ echo $(optix-install-dir)/SDK-src.build/lib/ptx ; }
optix-samples-scd(){ cd $(optix-samples-sdir) ; }
optix-samples-bcd(){ cd $(optix-samples-bdir) ; }
optix-samples-pcd(){ cd $(optix-samples-pdir) ; }

optix-samples-f64(){ ptx.py $(optix-samples-pdir) $* ; }



optix-samples--()
{
   optix-samples-setup
   optix-samples-cmake    
   optix-samples-make
}

optix-samples-notes(){ cat << EOI

   With CUDA_NVRTC_ENABLED=OFF in order to look at the PTX 
   find that had to copy one of the order 10 common.h up to 
   top level for it to be found.


EOI
}

optix-samples-info(){ cat << EOI

    optix-install-dir   : $(optix-install-dir)
    optix-samples-sdir  : $(optix-samples-sdir)
    optix-samples-bdir  : $(optix-samples-bdir)
    optix-samples-pdir  : $(optix-samples-pdir)

EOI
}

optix-samples-setup(){

   local msg="=== $FUNCNAME :"

   local iwd=$PWD
   local idir=$(optix-install-dir)
   local sdir=$(optix-samples-sdir)
   [ -d "$sdir" ] && echo $msg already setup in $sdir && return

   type $FUNCNAME

   optix-samples-info
   local ans
   read -p "$msg enter Y to continue "  ans
   [ "$ans" != "Y" ] && echo skip && return 


   cd $idir
   local cmd="$SUDO cp -R SDK SDK-src && $SUDO chown $USER SDK-src && $SUDO mkdir SDK-src.build && $SUDO chown $USER SDK-src.build "
   echo $cmd
   eval $cmd

   cd $iwd
}

optix-samples-clean(){
    local bdir=$(optix-samples-bdir)
    rm -rf $bdir   # starting clean 
}
optix-samples-cmake(){
    local iwd=$PWD
    local bdir=$(optix-samples-bdir)

    [ -f "$bdir/CMakeCache.txt" ] && echo $msg already configured && return 

    #rm -rf $bdir   # starting clean 
    mkdir -p $bdir
    cd $bdir

    cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) \
          -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
          -DCUDA_NVRTC_ENABLED=OFF \
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




optix-runfile-vers(){ echo ${OPTIX_RUNFILE_VERS:-600}  ; }
optix-runfile()
{
    case $(optix-runfile-vers) in
       510) echo NVIDIA-OptiX-SDK-5.1.0-linux64_24109458.sh ;;
       511) echo NVIDIA-OptiX-SDK-5.1.1-linux64-25109142.sh ;;
       600) echo NVIDIA-OptiX-SDK-6.0.0-linux64-25650775.sh ;;
    esac
}

optix-runfile-prefix-abs(){ echo $LOCAL_BASE/opticks/externals/OptiX_$(optix-runfile-vers) ; }
optix-runfile-prefix(){     echo $LOCAL_BASE/opticks/externals/OptiX ; }
optix-runfile-install()
{
    local msg="=== $FUNCNAME :"
    local iwd=$PWD

    local runfile=$(optix-runfile)
    [ ! -f "$runfile" ] && echo NO runfile $runfile in $PWD && return 

    local prefix=$(optix-runfile-prefix-abs)
    local name=$(basename $prefix)

    if [ -d "$prefix"  ]; then
        echo $msg name $name already installed to prefix $prefix  
    else
        echo $msg runfile $runfile 
        echo $msg name $name
        echo $msg prefix $prefix
        echo $msg Install using runfile ? 
        local ans  
        read -p "$msg You will need to enter \"y\" to accept the licence and then \"n\" to not include subdirectory name to the runfile installer. Press ENTER to proceed. " ans
        [ "$ans" != "" ] && echo abort && return

        mkdir -p $prefix
        sh $runfile --prefix=$prefix

        optix-runfile-link 
    fi

    cd $iwd
}

optix-runfile-link()
{
    local iwd=$PWD
    local absprefix=$(optix-runfile-prefix-abs)
    local absname=$(basename $absprefix)

    local stdprefix=$(optix-runfile-prefix)
    local stdname=$(basename $stdprefix)


    cd $(dirname $absprefix)
    [ ! -d $absname ] && echo $msg ERR no directory $absname && return 

    ln -svfn $absname $stdname

    cd $iwd
}




optix-runfile-install-notes(){ cat << EON
$FUNCNAME
=============================

optix-runfile-install
    run this from the directory in which you keep the runfile installers
    to install the OptiX SDK into a directory named after the version, such as
    $LOCAL_BASE/opticks/externals/OptiX_600

optix-runfile-link
    changes the symbolic link indicating which of the installed OptiX SDKs to use

    #OPTIX_RUNFILE_VERS=511 optix-runfile-link 
    OPTIX_RUNFILE_VERS=600 optix-runfile-link 

EON
}


optix-runfile-info(){ cat << EOI
$FUNCNAME
========================

   optix-runfile-vers        : $(optix-runfile-vers)
   optix-runfile             : $(optix-runfile)
   optix-runfile-prefix      : $(optix-runfile-prefix)
   optix-runfile-prefix-abs  : $(optix-runfile-prefix-abs)

   OPTIX_RUNFILE_VERS        : $OPTIX_RUNFILE_VERS 
   OPTICKS_OPTIX_INSTALL_DIR : $OPTICKS_OPTIX_INSTALL_DIR

   export OPTICKS_OPTIX_INSTALL_DIR=\$LOCAL_BASE/opticks/externals/optix

EOI
}



