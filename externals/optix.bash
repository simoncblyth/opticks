optix-src(){      echo externals/optix.bash ; }
optix-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(optix-src)} ; }
optix-vi(){       vi $(optix-source) ; }
optix-env(){      olocal- ; }
optix-usage(){ cat << EOU

NVIDIA OptiX Ray Trace Toolkit
================================== 

Resources
----------

* https://devtalk.nvidia.com/default/board/90/optix/
* http://docs.nvidia.com/gameworks/index.html#gameworkslibrary/optix/optix_programming_guide.htm


TODO: rearrange OptiX use to facilitate easier version switching
------------------------------------------------------------------

* avoid having to rebuild all of Opticks just to use a different OptiX version ?
* how far can forward decalarations of OptiX types in optixrap- get me
* http://stackoverflow.com/questions/3597693/how-does-the-pimpl-idiom-reduce-dependencies


OptiX Advanced Samples 
------------------------


* https://github.com/nvpro-samples/optix_advanced_samples
* https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixGlass
* https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixProgressivePhotonMap
* https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixVox


macrosim
-----------

* https://bitbucket.org/itom/macrosim/wiki/Home


Things that make createPTXFromFile prone to segv
----------------------------------------------------

* lots of rtPrintf, keep ALL rtPrintf inside ifdef so can easily switch them all off

* calling a function twice with 



RT_CALLABLE_PROGRAM (Extract from Manual)
--------------------------------------------

RT_CALLABLE_PROGRAMs can take arguments and return values just like other functions in CUDA, 
whereas RT_PROGRAMs must return void.

if you have a function that is invoked from many different places in your OptiX
node graph, making it an RT_CALLABLE_PROGRAM can reduce code replication and
compile time, and potentially improve runtime through increased warp
utilization.  

There are three pieces of callable programs. 

1. the program you wish to call. 
2. declaration of a proxy function used to call the callable program. 
3. host code used to associate a callable program with the proxy function 
   that will call it within the OptiX node graph.  

Callable programs come in two variants, bound and bindless. 

Bound programs 
    invoked by direct use of a program bound to a variable through the
    host API and inherit the semantic type and variable scope lookup 
    as the calling program. 

Bindless programs 
    are called via an ID obtained from the RTprogram on the host and unlike 
    bound programs do not inherit the semantic type or scope lookup of the calling program


OptiX 4.1.1
-------------

* https://developer.nvidia.com/designworks/optix/download

Graphics Driver : Linux: driver version 367.35 or later is required.

Operating System: Windows 7/8.1/10 64-bit; Linux RHEL 4.8+ or Ubuntu 10.10+ 64-bit; Mac OS 10.9 or higher

CUDA Toolkit: It is not required to have any CUDA toolkit installed in order to run OptiX-based applications.

CUDA Toolkit 6.5, 7.0, or 7.5: OptiX 4.0 has been built with CUDA 7.5, but any
specified toolkit should work when compiling PTX for OptiX. If an application
links against both the OptiX library and the CUDA runtime on Linux, it is
recommended to use the same version of CUDA that was used to build OptiX. OptiX
supports running with NVIDIA Parallel Nsight but does not currently support
kernel debugging. In addition, it is not recommended to compile PTX code using
any -G (debug) flags to nvcc.





PTX njuffa (2014)
--------------------

* https://devtalk.nvidia.com/default/topic/788577/?comment=4365842


PTX is a virtual instruction set that exposes little beyond instructions
supported by GPU hardware. There are some exceptions for operations that are
commonly present as instructions on other compute platforms, such as integer
and floating-point division which are instructions at the PTX level, but really
implemented as emulation routines "under the hood".

GPU hardware provides minimal hardware support for the following higher
single-precision operations: reciprocal, reciprocal square root, sine, cosine,
exponentiation base 2, logarithm base 2. These are exposed via PTX. CUDA offers
some device function intrinsics [such as__sinf(), __cosf()] which are thin
wrappers around these PTX instructions. If CUDA code is built with
-use_fast_math, some math library functions [such as sinf() and cosf()] are
mapped automatically to the corresponding intrinsic. From your description
above it sound slike this is how you may be building your code?

You can find the supported PTX instructions in the document ptx_isa_4.1.pdf
that ships with CUDA. For your purposes, you would want to consult section
8.7.3 Floating-point instructions. For example, the PTX instruction "sin" is
described in sub-section 8.7.3.18 with the following synopsis:

sin.approx{.ftz}.f32 d, a;

As can be seen, there is no double-precision version of this instruction (since
no such hardware instruction exists in the GPU).

Generally, the single-precision hardware implementations mentioned above are
very high performance but "quick & dirty" since they were designed for use in
graphics. Comprehensive math libraries for general computation obviously
require many more functions and also typically need higher accuracy and better
special case handling as prescribed by the IEEE-754 floating-point standard and
the ISO C/C+ standards. Note also that the hardware does not provide any kind
of higher double-precision operations.

Like just about any other computing platform including x86 and ARM, CUDA
therefore ships with a math library that sits on top of the assembly language
level (i.e. upstream of PTX) in the software stack. In CUDA 6.5, the math
library is provided as part of a device library. The documentation for this
device library resides in a file called libdevice-users-guide.pdf that ships
with CUDA. The actual code is in multiple files libdevice.compute_??.??.bc.
Best I know these libraries are usable by tool chains other than CUDA and I
believe there is at least one project which makes use of that.

Here is a presentation from GTC 2013 that shows how GPU compilers are
structured. On slide 11 it is shown where the contents of libdevice enters the
flow inside the tool chain, well before the PTX assembly code is generated:

http://on-demand.gputechconf.com/gtc/2013/presentations/S3185-Building-GPU-Compilers-libNVVM.pdf


With CUDA 7.0 via soft link /Developer/NVIDIA/CUDA-7.0/doc

* /usr/local/cuda/doc/pdf/ptx_isa_4.2.pdf
* /usr/local/cuda/doc/pdf/libdevice-users-guide.pdf



* https://devtalk.nvidia.com/default/topic/788577/?comment=4365842

Single-precision math functions sinf() and cosf() normally map [like
double-precision sin(), cos()] to functions in libdevice. But when you compile
with -use_fast_math, they are replaced by the intrinsics __sinf() and __cosf(),
which map directly to the PTX instructions `sin.ftz.f32` and `cos.ftz.f32`. As
a consequence the resulting PTX would compile even without libdevice being part
of the build.

[Later:] As for MADC.HI, check your architecture target and PTX version number.
My memory is hazy, but this instruction was only added three years ago or so.
How the error relates to -use_fast_math, I have no clue.


* 


Please always provide the OptiX version you're having issues with.

There have been fixes around the use of madc.hi inside OptiX 3.6.3, see 
https://devtalk.nvidia.com/default/topic/779816/optix/optix-3-6-3-released/
"Bug fix Double-precision cosine and other transcendentals now work properly, although ray tracing internals are still single-precision."

If that doesnt' help, please provide a minimal reproducer in failing state to the OptiX team including the following information:
OS version, OS bitness, OptiX version, CUDA toolkit version, installed GPU(s), display driver version.




* https://devtalk.nvidia.com/default/topic/780372/?comment=4330651

If I don't use -use_fast_math while generating .ptx files, I'll have runtime
error: OptiX Error: Invalid value (Details: Function "RTresult
_rtProgramCreateFromPTXFile(RTcontext, const char*, const char*,
RTprogram_api**)" caught exception: defs/uses not defined for PTX instruction
(Try --use_fast_math): madc.hi, [1310866]) (sample2.cxx:219)

If I use -ftz -prec-div -prec-sqrt -fmad instead, I got the same error as using
nothing.  Even if I use -ftz=true -prec-div=false -prec-sqrt=false -fmad=true
to get the same effect as -use_fast_math, I got the same error. 

Any ideas?  Thanks!



Detlef Roettger (2016)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/930762/?comment=4857695

OptiX doesn't support function call instructions inside PTX code unless they
have been created via OptiX callable program mechanism.  Other function calls
are always inlined. To make sure that happens in the CUDA to PTX compilation
process already I'm declaring all functions in OptiX CUDA C++ with
__forceinline__ __device__

__inline__ alone is not enough, because that's just a hint and the CUDA
compiler might decide to not inline a function if it's too big or has too many
arguments. I've seen that happening.

I'm using a define for that which looks like this:

#ifndef RT_FUNCTION
#define RT_FUNCTION __forceinline__ __device__
#endif

// Example: 
RT_FUNCTION int random()
{
  return 4; // Random number, determined by fair dice roll! ;-)
}

Depending on how you use the callable program (bound to a variable or bindless
via ID) there are also different scopes you can access!  Bound callable
programs inherit the scope of the caller, bindless callable programs only have
the context and themselves, the program, as scope. Means bindless callable
programs cannot call rtTrace() or rtTransform*() functions for example.  Refer
to the OptiX Programming Guide for more information.



Refs describing GPU architecture evolution
--------------------------------------------

* https://cryptojedi.org/papers/gpus-20130310.pdf


Pre-CUDA survey of GPGPU from 2005

* https://research.nvidia.com/sites/default/files/publications/ASurveyofGeneralPurposeComputationonGraphicsHardware.pdf


* http://disi.unal.edu.co/~gjhernandezp/HeterParallComp/GPU/gpu-hist-paper.pdf


A Review of CUDA, MapReduce, and Pthreads Parallel Computing Models

* https://arxiv.org/pdf/1410.4453.pdf


Kepler white paper

* http://www.nvidia.com/content/PDF/product-specifications/GeForce_GTX_680_Whitepaper_FINAL.pdf


* https://www.researchgate.net/publication/301363311_Architectural_evolution_of_NVIDIA_GPUs_for_High-Performance_Computing


NVIDIA Research
-----------------

* https://research.nvidia.com/nvpeople


Inline PTX ASM
---------------

* https://devtalk.nvidia.com/default/topic/452960/asm-inlining-in-cuda-code-/

::

    simon:include blyth$ optix-find asm
    simon:include blyth$ optix-ifind asm
    /Developer/OptiX/include/internal/optix_internal.h:    asm volatile("call (%0, %1, %2), _rt_texture_get_size_id, (%3);" :
    /Developer/OptiX/include/internal/optix_internal.h:    asm volatile("call (%0, %1, %2, %3), _rt_texture_get_gather_id, (%4, %5, %6, %7, %8);" :
    /Developer/OptiX/include/internal/optix_internal.h:    asm volatile("call (%0, %1, %2, %3), _rt_texture_get_base_id, (%4, %5, %6, %7, %8, %9);" :


* https://github.com/facebook/fbcuda/blob/master/CudaUtils.cuh


Pointer Arithmetic Not Allowed on OptiX buffers
-------------------------------------------------

* https://devtalk.nvidia.com/default/topic/872983/optix/directx-gt-optix-single-geometry-buffer-or-multiple-/

Detlef Roettger:

    The only way to access a single buffer element, whatever that is, including a
    user defined struct, is the operator[]! Pointer arithmetic is not allowed on
    buffers!!!


PyOptiX
--------

* https://github.com/ozen/PyOptiX


Understanding the efficiency of ray traversal on GPUs
-------------------------------------------------------

* :google:`Understanding the Efficiency of Ray Traversal on GPUs – Kepler and Fermi Addendum`

* https://research.nvidia.com/users/timo-aila
* https://research.nvidia.com/users/samuli-laine
* https://research.nvidia.com/users/tero-karras

* https://code.google.com/archive/p/understanding-the-efficiency-of-ray-traversal-on-gpus/
* https://mediatech.aalto.fi/~samuli/publications/aila2012tr1_poster.pdf


CG courses
------------

Indepth covering the "Understanding.." paper.

* http://www.cs.uu.nl/docs/vakken/magr/2015-2016/


* http://www.cs.uu.nl/docs/vakken/magr/2015-2016/slides/lecture%2002%20-%20acceleration%20structures.pdf

* http://www.cs.uu.nl/docs/vakken/magr/2015-2016/slides/lecture%2011%20-%20GPU%20ray%20tracing%20%281%29.pdf
* http://www.cs.uu.nl/docs/vakken/magr/2015-2016/slides/lecture%2012%20-%20GPU%20ray%20tracing%20(2).pdf


CUDA Raytrace Thesis
----------------------

* http://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1024&context=techmasters

* ~/opticks_refs/andrew_britton_cuda_raytrace_thesis_purdue.pdf



OptiX Thread Timing
----------------------

* https://devtalk.nvidia.com/default/topic/980319/optix/timing-and-profiling-with-optix/

::

    clock_t start_time = clock();

    // some small amount of code

    clock_t stop_time = clock();
        
    int time = (int)(stop_time - start_time);
    rtPrintf("time in func %fms\n", time / clockRate);

::

    simon:SDK blyth$ optix-find clock
    /Developer/OptiX/SDK/optixBuffersOfBuffers/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixBuffersOfBuffers/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixCallablePrograms/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixCallablePrograms/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixConsole/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixConsole/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixDynamicGeometry/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixDynamicGeometry/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixInstancing/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixInstancing/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixMeshViewer/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixMeshViewer/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixPrimitiveIndexOffsets/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixPrimitiveIndexOffsets/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixProgressiveVCA/camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixProgressiveVCA/camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixSelector/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixSelector/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixSphere/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixSphere/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixSpherePP/pinhole_camera.cu:  clock_t t0 = clock(); 
    /Developer/OptiX/SDK/optixSpherePP/pinhole_camera.cu:  clock_t t1 = clock(); 
    /Developer/OptiX/SDK/optixDeviceQuery/optixDeviceQuery.cpp:            int clock_rate;
    /Developer/OptiX/SDK/optixDeviceQuery/optixDeviceQuery.cpp:            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(clock_rate), &clock_rate));
    /Developer/OptiX/SDK/optixDeviceQuery/optixDeviceQuery.cpp:            printf("  Clock Rate: %u kilohertz\n", clock_rate);
    /Developer/OptiX/SDK/support/freeglut/include/GL/freeglut_ext.h: * NB: front facing polygons have clockwise winding, not counter clockwise
    /Developer/OptiX/SDK/support/freeglut/include/GL/freeglut_std.h: * NB: front facing polygons have clockwise winding, not counter clockwise
    simon:SDK blyth$ 




CUDAraster
----------

* https://code.google.com/archive/p/cudaraster/

High-Performance Software Rasterization on GPUs
Samuli Laine and Tero Karras,
Proc. High-Performance Graphics 2011
http://research.nvidia.com/sites/default/files/publications/laine2011hpg_paper.pdf


Alternatives to OptiX
-----------------------

* http://raytracey.blogspot.tw/2016/05/start-your-engines-source-code-for.html


Rayforce
~~~~~~~~~~~

* http://rayforce.survice.com
* https://gitlab.survice.com/survice/Rayforce/tree/master

Ray tracer referenced from Multi-Hit Ray Traversal paper

* http://jcgt.org/published/0003/01/01/paper.pdf


Intel : Embree  (works on CPUs and many integrated core (MICs) coprocessors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://neams.rpi.edu/jiw2/papers/M&C2015%20Liu02.pdf

AMD : RadeonRays, FireRays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://gpuopen.com/firerays-2-0-open-sourcing-and-customizing-ray-tracing/
* https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK
* https://m.reddit.com/r/Amd/comments/4p4t9y/gpuopen_firerays/

PowerVR Wizard GPUs : raycast specialized hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://imgtec.com/blog/gdc-2016-ray-tracing-graphics-mobile/
* https://imgtec.com/blog/real-time-ray-tracing-on-powervr-gr6500-ces-2016/


Fora Posts
------------

* https://devtalk.nvidia.com/default/topic/815975/optix/geometryinstances-toggling-objects-between-visible-invisible/
* https://devtalk.nvidia.com/default/topic/817572/optix/graph-nodes-in-optix/
* https://devtalk.nvidia.com/default/topic/771998/optix/optix-acceleration-structure/


Moving Geometry
~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/952332/optix/move-objects/


Launch Times : removeVariable triggers megakernel recompile, avoid leaking buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/938033/optix/excessive-setup-times/

Separate Thread OptiX
~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/938478/optix/is-it-possible-for-optix-to-run-in-a-background-host-thread/

Detlef:

Running OptiX in a separate thread should just work.
Just make sure only that thread is doing OptiX calls. 
The OptiX API is not guaranteed to be multi-threaded safe.

Doing this will not improve the renderer time.  Also if the GUI is run on the
same GPU as OptiX there will always be a sluggishness to the GUI unless your
OptiX renderer has a very small runtime per frame.  With a dedicated compute
GPU or VCA cluster, running OptiX fully asynchronous in a separate host thread
would make the GUI rendering independent of the ray tracing. That's one thing
on my list to implement in a renderer app.


Dirty CUDA Interop Buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/925622/optix/should-i-free-buffer-data-/

Detlef:

Another case would be CUDA interop buffers which use device side pointers where
the update happens through CUDA device code. Then you'd need to make the buffer
dirty manually to let OptiX know its contents have changed, to be able to
rebuild accelerations structures etc.)



Downloading OptiX
-------------------

Login as NVIDIA registered developer, then 

* https://developer.nvidia.com/designworks
* https://developer.nvidia.com/designworks/optix/
* https://developer.nvidia.com/designworks/optix/downloads/legacy


Delete the symbolic link before exploding the downloaded package installer::

    simon:opticks blyth$ l /Developer/
    total 8
    lrwxr-xr-x  1 root  wheel    9 Jun 29  2015 OptiX -> OptiX_380
    drwxr-xr-x  4 root  wheel  136 Jun 29  2015 NVIDIA
    drwxr-xr-x  7 root  admin  238 May 29  2015 OptiX_380
    drwxr-xr-x  7 root  admin  238 Jan 22  2015 OptiX_301
    drwxr-xr-x  7 root  admin  238 Dec 18  2014 OptiX_370b2


    simon:Developer blyth$ sudo rm OptiX






Replace the symbolic link afterwards::

    simon:Developer blyth$ sudo mv OptiX OptiX_400
    simon:Developer blyth$ sudo ln -s OptiX_400 OptiX
    simon:Developer blyth$ l
    total 8
    lrwxr-xr-x  1 root  wheel    9 Aug 10 14:42 OptiX -> OptiX_400
    drwxr-xr-x  8 root  admin  272 Jul 22 05:17 OptiX_400
    drwxr-xr-x  4 root  wheel  136 Jun 29  2015 NVIDIA
    drwxr-xr-x  7 root  admin  238 May 29  2015 OptiX_380
    drwxr-xr-x  7 root  admin  238 Jan 22  2015 OptiX_301
    drwxr-xr-x  7 root  admin  238 Dec 18  2014 OptiX_370b2


Check 411 on Mac
~~~~~~~~~~~~~~~~~~~

::

    simon:Developer blyth$ sudo rm OptiX

    open /Users/blyth/Downloads/NVIDIA-OptiX-SDK-4.1.1-mac64-22553582.dmg  # GUI installer

    simon:Developer blyth$ sudo mv OptiX OptiX_411
    simon:Developer blyth$ sudo ln -s OptiX_380 OptiX




Try samples
-------------

::

    simon:SDK-precompiled-samples blyth$ open optixDynamicGeometry.app
    LSOpenURLsWithRole() failed with error -10810 for the file /Developer/OptiX_400/SDK-precompiled-samples/optixDynamicGeometry.app.
    simon:SDK-precompiled-samples blyth$ open .
    simon:SDK-precompiled-samples blyth$ open optixTutorial.app
    LSOpenURLsWithRole() failed with error -10810 for the file /Developer/OptiX_400/SDK-precompiled-samples/optixTutorial.app.
    simon:SDK-precompiled-samples blyth$ 



    sudo launchctl stop com.apple.security.syspolicy 
    sudo launchctl start com.apple.security.syspolicy 


Stopping com.apple.security.syspolicy doesnt fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some launch permissions issue ?::

    simon:SDK-precompiled-samples blyth$ sudo launchctl stop com.apple.security.syspolicy
    Password:
    simon:SDK-precompiled-samples blyth$ open optixMeshViewer.app
    LSOpenURLsWithRole() failed with error -10810 for the file /Developer/OptiX_411/SDK-precompiled-samples/optixMeshViewer.app.
    simon:SDK-precompiled-samples blyth$ sudo launchctl start com.apple.security.syspolicy
    simon:SDK-precompiled-samples blyth$ open optixMeshViewer.app
    LSOpenURLsWithRole() failed with error -10810 for the file /Developer/OptiX_411/SDK-precompiled-samples/optixMeshViewer.app.
    simon:SDK-precompiled-samples blyth$ 


try building 411 samples
~~~~~~~~~~~~~~~~~~~~~~~~~


::

    simon:build blyth$ which nvcc
    simon:build blyth$ cuda-
    simon:build blyth$ which nvcc
    /Developer/NVIDIA/CUDA-7.0/bin/nvcc
    simon:build blyth$ 
    simon:build blyth$ 
    simon:build blyth$ cmake ../SDK
    -- The C compiler identification is AppleClang 6.0.0.6000057
    -- The CXX compiler identification is AppleClang 6.0.0.6000057
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Checking to see if CXX compiler accepts flag -Wno-unused-result
    -- Checking to see if CXX compiler accepts flag -Wno-unused-result - yes
    -- Performing Test SSE_41_AVAILABLE
    -- Performing Test SSE_41_AVAILABLE - Success
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Looking for pthread_create
    -- Looking for pthread_create - found
    -- Found Threads: TRUE  
    -- Found CUDA: /Developer/NVIDIA/CUDA-7.0 (found suitable version "7.0", minimum required is "5.0") 
    -- Found OpenGL: /System/Library/Frameworks/OpenGL.framework  
    -- Found GLUT: /System/Library/Frameworks/GLUT.framework  
    -- Configuring done
    CMake Warning (dev):
      Policy CMP0042 is not set: MACOSX_RPATH is enabled by default.  Run "cmake
      --help-policy CMP0042" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      MACOSX_RPATH is not specified for the following targets:

       sutil_sdk

    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Generating done
    -- Build files have been written to: /Developer/OptiX/build




Some succeed others giving ptx assembly aborted::


    simon:bin blyth$ ./optixDynamicGeometry 
    Using multi-acceleration mode
    Creating geometry ... done
    Validating ... done
    Preprocessing scene ... OptiX Error: 'Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: ptxas application ptx input, line 504; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 515; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 526; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 537; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 582; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 595; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 608; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 621; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1019; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1020; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1021; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1023; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1053; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1055; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1057; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1059; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1229; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1231; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1233; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1235; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas fatal   : Ptx assembly aborted due to errors returned (209): No binary for GPU)'

    simon:bin blyth$ ./optixPathTracer 
    OptiX Error: 'Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: ptxas application ptx input, line 250; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 253; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 255; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 257; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 291; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 296; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 301; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 306; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas fatal   : Ptx assembly aborted due to errors returned (209): No binary for GPU)'
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error
    Abort trap: 6
    simon:bin blyth$ 



talonmies
~~~~~~~~~~~

* https://stackoverflow.com/questions/15168965/cuda-error-cuda-error-no-binary-for-gpu

Try to compile the PTX with the toolchain yourself::

    $ ptxas -arch=sm_20 own.ptx 
    ptxas own.ptx, line 24; error   : Arguments mismatch for instruction 'mov'
    ptxas own.ptx, line 24; error   : Unknown symbol 'func_retval0'
    ptxas own.ptx, line 24; error   : Label expected for forward reference of 'func_retval0'
    ptxas fatal   : Ptx assembly aborted due to errors




OptiX 4.1.1 (Aug 2017)
------------------------

* https://developer.nvidia.com/designworks/optix/download





OptiX 4 Released : 07/26/2016
-------------------------------

* https://devtalk.nvidia.com/default/topic/952430/optix/optix-4-0-is-released-/
* https://devtalk.nvidia.com/default/topic/950165/optix/looking-for-optix-4-adopters/

Today we are happy to announce the release of OptiX 4.0.

This version is an important milestone in the evolution of OptiX, featuring a
complete re-implementation of many core components, including an all-new
LLVM-based compilation pipeline. The internal redesign has been in the works
for several years, and lays the groundwork for better overall performance,
better multi-GPU scaling, better debugging and profiling, and many other
exciting features. Version 4.0 maintains compatibility with your existing
applications and provides the same easy to use API for which OptiX is known.

You can find the download and release notes at:
http://developer.nvidia.com/optix

OptiX 4 Release Notes
-----------------------

* https://developer.nvidia.com/designworks/optix/download

CUDA Toolkit 6.5, 7.0, or 7.5: 

OptiX 4.0 has been built with CUDA 7.5, but any
specified toolkit should work when compiling PTX for OptiX. If an application
links against both the OptiX library and the CUDA runtime on Linux, it is
recommended to use the same version of CUDA that was used to build OptiX. OptiX
supports running with NVIDIA Parallel Nsight but does not currently support
kernel debugging. In addition, it is not recommended to compile PTX code using
any -G (debug) flags to nvcc.


OptiX 4 Enabled Exceptions Are Expensive
-------------------------------------------

* https://devtalk.nvidia.com/default/topic/952532/optix/optix-4-0-runs-slow/

Mentions exceptions are very costly in 400.

From pdf p45:

By default, only RT_EXCEPTION_STACK_OVERFLOW is enabled. During debugging, it
is often useful to turn on all available exceptions. This can be achieved with
a single call:

::

    rtContextSetExceptionEnabled(context, RT_EXCEPTION_ALL, 1);



FIXED : OptiX 4 Issues : Textures
------------------------------------

OptiX 4 not working with old OPropertyLib texture configuration, suspect it is being more strict.
To understand OptiX textures need to know the background from OpenGL and CUDA.

* https://www.opengl.org/wiki/Sampler_Object
* https://www.opengl.org/wiki/Texture_Storage
* https://open.gl/textures
* https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
* http://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf


Fixed by moving to normalized float addressing of textures.


OptiX 4 Buffer Issues
-------------------------

* https://devtalk.nvidia.com/default/topic/946870/optix/optix-4-and-cuda-interop-new-limitation-with-input-output-buffers/


OptiX 4.0 Beta 
----------------

* does not support Mac
* drops Fermi
* https://devtalk.nvidia.com/default/topic/939598/optix/running-optix-4-0beta-with-cuda-8-0rc-not-supported-/

Just use CUDA 7.5 for now.
The generated PTX code is rewritten by OptiX internally and sent to the CUDA driver for assembly. 
That CUDA driver comes with the display driver and will support the Pascal architecture.
Your original PTX input doesn't need to be compiled for the newest Streaming Multiprocessor version.
Anything from SM 2.0 to SM 5.2 will do with CUDA 7.5.




OptiX 3.9.1 (June 2016)
---------------------------

* https://developer.nvidia.com/designworks/optix/download

* Added support for Pascal GPU architectures.
* Improved performance with large node graphs.
* Improved compile times.


OptiX 3.9 (December 2015)
--------------------------

* https://devtalk.nvidia.com/default/topic/903433/optix/optix-3-9-is-released-/

With this release, we're also introducing a new download mechanism and you'll
be able to download the SDK directly from our website after joining the
DesignWorks developer program and completing a short survey. You will no longer
need to use the FTP site to download OptiX.

* https://devtalk.nvidia.com/default/topic/939304/optix/running-optix-on-cpu/

  * updating to 3.9 is advised
Release Notes
~~~~~~~~~~~~~~~~

Hardware 
    CUDA capable devices of Compute Capability 2.0 (“Fermi”) or higher are supported on GeForce, Quadro, or Tesla

Driver
    The CUDA R346 or later driver is required. For the Mac, the driver extension
    module supplied with CUDA 7.5 will need to be installed.

OS
    Windows 7/8/8.1/10 64-bit; Linux RHEL 4.8+ or Ubuntu 10.10+ - 64-bit; MacOS
    10.9+ Note: 32-bit operating systems are no longer supported.

CUDA Toolkit 4.0 – 7.5.
    OptiX 3.9 has been built with CUDA 7.5, but any specified toolkit should work
    when compiling PTX for OptiX. If an application links against both the OptiX
    library and the CUDA runtime on Linux, it is recommended to use the same
    version of CUDA that was used to build OptiX.

C/C++ Compiler
    Visual Studio 2008, 2010, 2012, or 2013 is required on Windows systems. 
    gcc 4.4-4.8 have been tested on Linux. 
    Xcode 6 has been tested on Mac OSX 10.9. 
    See the CUDA Toolkit documentation for more information on supported compilers.



Observations
~~~~~~~~~~~~~~

* currently on CUDA 7.0
* no windows MINGW ? 




Conference Talks
-------------------

* http://www.nvidia.com/object/siggraph2015-best-gtc.html

* http://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php?searchByKeyword=Optix&searchItems=&sessionTopic=&sessionEvent=&sessionYear=&sessionFormat=&submit=&select=+

  Many presentations (videos and pdfs) on OptiX


Approximate Global Illumination using Voxel Cone Tracing, VXGI
-----------------------------------------------------------------

See vxgi-


FindOptiX.cmake
----------------

::

    simon:~ blyth$ mdfind -name FindOptiX.cmake
    /Developer/OptiX_301/SDK/CMake/FindOptiX.cmake
    /Developer/OptiX_370b2/SDK/CMake/FindOptiX.cmake
    /usr/local/env/cuda/OptiX_370b2_sdk/CMake/FindOptiX.cmake
    /usr/local/env/cuda/OptiX_380_sdk/CMake/FindOptiX.cmake
    /Developer/OptiX_380/SDK/CMake/FindOptiX.cmake
    /usr/local/env/graphics/hrt/cmake/Modules/FindOptix.cmake
    /usr/local/env/graphics/photonmap/CMake/FindOptiX.cmake
    /usr/local/env/optix/macrosim/macrosim_tracer/CMake/FindOptiX.cmake



OptiX Release Notes regards multi-GPU and SLI
-----------------------------------------------

* https://en.wikipedia.org/wiki/Scalable_Link_Interface

SLI is a multi-GPU technology developed by NVIDIA for linking two or more video
cards together to produce a single output


SLI is not required for OptiX to use multiple GPUs, and it interferes when
OptiX uses either D3D or OpenGL interop. Disabling SLI will not degrade OptiX
performance and will provide a more stable environment for OptiX applications
to run. SLI is termed “Multi-GPU mode” in recent NVIDIA Control Panels, with
the correct option being “Disable multi-GPU mode” to enable OptiX to freely use
all system GPUs.

::

   SLI looks to cause significant complications



Excellent Comprehensive Slides on Ray Tracing
------------------------------------------------

* https://mediatech.aalto.fi/~jaakko/ME-C3100/S2013/12_RayTracing.pdf
* https://mediatech.aalto.fi/~jaakko/ME-C3100/S2013/


Samples
--------

julia : example of analytic geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/usr/local/env/cuda/OptiX_380_sdk/julia/julia.cpp::

    414 void JuliaScene::createGeometry()
    415 {
    416   // Julia object
    417   Geometry julia = m_context->createGeometry();
    418   julia->setPrimitiveCount( 1u );
    419   julia->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "julia", "julia.cu" ), "bounds" ) );
    420   julia->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "julia", "julia.cu" ), "intersect" ) );
    421 

::

    167 RT_PROGRAM void intersect(int primIdx)
    ///
    ///    primIdx not used
    ///
    168 {
    169   float tmin, tmax;
    170   if( intersectBoundingSphere(ray.origin, ray.direction, tmin, tmax) )
    171   {
    172     JuliaSet distance( max_iterations );


progressivePhotonMap
~~~~~~~~~~~~~~~~~~~~~~~

Stages:

* trace viewing rays from camera position thru pixels into scene
  collecting rtBuffer<HitRecord,2> rtpass_output_buffer
  which contain hit position, normal and a starting radius and 
  initialize stats to zero for this "region"/pixel   

  This initial stage only need to be repeated when camera position changes, 
  it is identifying surfaces relevant to the current view.

* 





Note implicit "union" between PhotonRecord and PackedPhotonRecord

/usr/local/env/cuda/OptiX_380_sdk/progressivePhotonMap/ppm.h::

    089 struct PhotonRecord
     90 {
     91   optix::float3 position;
     92   optix::float3 normal;      // Pack this into 4 bytes
     93   optix::float3 ray_dir;
     94   optix::float3 energy;
     95   optix::uint   axis;
     96   optix::float3 pad;
     97 };
     98 
     99 
    100 struct PackedPhotonRecord
    101 {
    102   optix::float4 a;   // position.x, position.y, position.z, normal.x
    103   optix::float4 b;   // normal.y,   normal.z,   ray_dir.x,  ray_dir.y
    104   optix::float4 c;   // ray_dir.z,  energy.x,   energy.y,   energy.z
    105   optix::float4 d;   // axis,       padding,    padding,    padding
    106 };


/usr/local/env/cuda/OptiX_380_sdk/progressivePhotonMap/ppm.cpp::

    140   enum ProgramEnum {
    141     rtpass,
    142     ppass,
    143     gather,
    144     numPrograms
    145   };
    ///
    /// 3 stage    
    /// 
    572 void ProgressivePhotonScene::trace( const RayGenCameraData& camera_data )
    580   if ( m_camera_changed ) {
    587      // Trace viewing rays
    592      m_context->launch( rtpass,
    593                       static_cast<unsigned int>(buffer_width),
    594                       static_cast<unsigned int>(buffer_height) );
    597     m_context["total_emitted"]->setFloat(  0.0f );
    598     m_iteration_count=1;
    599   }

    601   // Trace photons
    ///      trace photons from lights, and collect non-specular non-direct records
    610   m_context->launch( ppass,
    611                     static_cast<unsigned int>(m_photon_launch_width),
    612                     static_cast<unsigned int>(m_photon_launch_height) );
    ...
    621   // Build KD tree 
    624   createPhotonMap();
    ...
    628   // Shade view rays by gathering photons
    631   m_context->launch( gather,
    632                     static_cast<unsigned int>(buffer_width),
    633                     static_cast<unsigned int>(buffer_height) );


ppm_rtpass.cu::

    /// camera ray launch, collecting HitRecords for diffuse surface intersections

    039 rtBuffer<HitRecord, 2>           rtpass_output_buffer;
    ...
     89 RT_PROGRAM void rtpass_closest_hit()
     90 {
    ...
    104   float3 hit_point    = origin + t_hit*direction;
    105 
    106   if( fmaxf( Kd ) > 0.0f ) {
    107     // We hit a diffuse surface; record hit and return
    108     HitRecord rec;
    109     rec.position = hit_point;
    110     rec.normal = ffnormal;
    ...
    125     rec.flags = PPM_HIT;
    127     rec.radius2 = rtpass_default_radius2;
    128     rec.photon_count = 0;
    129     rec.accum_atten = 0.0f;
    130     rec.flux = make_float3(0.0f, 0.0f, 0.0f);
    131    
    132     rtpass_output_buffer[launch_index] = rec;
    133   } else {
    134     // Make reflection ray
    135     hit_prd.attenuation = hit_prd.attenuation * Ks;
    136     hit_prd.ray_depth++;
    137     float3 R = reflect( direction, ffnormal );
    138     optix::Ray refl_ray( hit_point, R, rtpass_ray_type, scene_epsilon );
    139     rtTrace( top_object, refl_ray, hit_prd );
    140   }



ppm_ppass.cu::

    040 rtBuffer<PhotonRecord, 1>        ppass_output_buffer;    
    ...
     94 RT_PROGRAM void ppass_camera()        
     95 {                                     
     96   size_t2 size     = photon_rnd_seeds.size();
     97   uint    pm_index = (launch_index.y * size.x + launch_index.x) * max_photon_count;
     98   uint2   seed     = photon_rnd_seeds[launch_index]; // No need to reset since we dont reuse this seed
     99 
    100   float2 direction_sample = make_float2(
    101       ( static_cast<float>( launch_index.x ) + rnd( seed.x ) ) / static_cast<float>( size.x ),
    102       ( static_cast<float>( launch_index.y ) + rnd( seed.y ) ) / static_cast<float>( size.y ) );
    ///
    ///     divvy up 0. -> 1. corresponding to the launch index with random offset within the divvied region
    ///     and use that for sampling portions of the light 
    ///
    103   float3 ray_origin, ray_direction;
    104   if( light.is_area_light ) {
    105     generateAreaLightPhoton( light, direction_sample, ray_origin, ray_direction );
    106   } else {
    107     generateSpotLightPhoton( light, direction_sample, ray_origin, ray_direction );
    108   }
        ...
    110   optix::Ray ray(ray_origin, ray_direction, ppass_and_gather_ray_type, scene_epsilon );
    111 
    112   // Initialize our photons
    113   for(unsigned int i = 0; i < max_photon_count; ++i) {
    114     ppass_output_buffer[i+pm_index].energy = make_float3(0.0f);
    115   }
    116 
    117   PhotonPRD prd;
    118   //  rec.ray_dir = ray_direction; // set in ppass_closest_hit
    119   prd.energy = light.power;
    120   prd.sample = seed;
    121   prd.pm_index = pm_index;
    122   prd.num_deposits = 0;
    123   prd.ray_depth = 0;
    124   rtTrace( top_object, ray, prd );
    125 }




    139 RT_PROGRAM void ppass_closest_hit()
    ///
    ///    keep bouncing and recording diffuse intersection PhotonRec 
    ///    after 1st : max_photon_count is only 2 
    ///
    ///
    140 {
    ...
    146   float3 hit_point = ray.origin + t_hit*ray.direction;
    147   float3 new_ray_dir;
    148 
    149   if( fmaxf( Kd ) > 0.0f ) {
    150     // We hit a diffuse surface; record hit if it has bounced at least once
    151     if( hit_record.ray_depth > 0 ) {
    152       PhotonRecord& rec = ppass_output_buffer[hit_record.pm_index + hit_record.num_deposits];
    153       rec.position = hit_point;
    ...   
    157       hit_record.num_deposits++;
    158     }
    159 
    160     hit_record.energy = Kd * hit_record.energy;
    ...
    163     sampleUnitHemisphere(rnd_from_uint2(hit_record.sample), U, V, W, new_ray_dir);
    164 
    165   } else {
    166     hit_record.energy = Ks * hit_record.energy;
    168     new_ray_dir = reflect( ray.direction, ffnormal );
    169   }
    ///
    ///    attenuate the energy (which starts as characteristic float3 light.power)
    ///    at each bounce by the Kd or Ks characteristic of the surface
    ///
    170 
    171   hit_record.ray_depth++;
    172   if ( hit_record.num_deposits >= max_photon_count || hit_record.ray_depth >= max_depth)
    173     return;
    174 
    175   optix::Ray new_ray( hit_point, new_ray_dir, ppass_and_gather_ray_type, scene_epsilon );
    176   rtTrace(top_object, new_ray, hit_record);
    177 }


ppm.cpp::

    219 void ProgressivePhotonScene::initScene( InitialCameraData& camera_data )
    ...
    302   // Photon pass
    303   m_photons = m_context->createBuffer( RT_BUFFER_OUTPUT );
    304   m_photons->setFormat( RT_FORMAT_USER );
    305   m_photons->setElementSize( sizeof( PhotonRecord ) );
    306   m_photons->setSize( m_num_photons );
    307   m_context["ppass_output_buffer"]->set( m_photons );
    ...
    ... 
    525 void ProgressivePhotonScene::createPhotonMap()
    526 {
    527   PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( m_photons->map() );
    528   PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>( m_photon_map->map() );
    ...
    534   // Push all valid photons to front of list
    535   unsigned int valid_photons = 0;
    536   PhotonRecord** temp_photons = new PhotonRecord*[m_num_photons];
    537   for( unsigned int i = 0; i < m_num_photons; ++i ) {
    538     if( fmaxf( photons_data[i].energy ) > 0.0f ) {
    539       temp_photons[valid_photons++] = &photons_data[i];
    540     }
    541   }
    ...
    548   // Make sure we aren't at most 1 less than power of 2
    549   valid_photons = valid_photons >= m_photon_map_size ? m_photon_map_size : valid_photons;
    ...
    564   // Now build KD tree
    ///
    ///        starts with all valid photons
    ///        these are recursively spatially partitioned 
    ///
    565   buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, m_split_choice, bbmin, bbmax );
    566 
    567   delete[] temp_photons;
    568   m_photon_map->unmap();
    569   m_photons->unmap();
    570 }
    ...
    ...
    412 void buildKDTree( PhotonRecord** photons, int start, int end, int depth, PhotonRecord* kd_tree, int current_root,
    413                   SplitChoice split_choice, float3 bbmin, float3 bbmax)
    414 {
    415   // If we have zero photons, this is a NULL node
    416   if( end - start == 0 ) {   
    417     kd_tree[current_root].axis = PPM_NULL;
    418     kd_tree[current_root].energy = make_float3( 0.0f );
    419     return;
    420   }
    421 
    422   // If we have a single photon
    423   if( end - start == 1 ) {
    424     photons[start]->axis = PPM_LEAF;
    425     kd_tree[current_root] = *(photons[start]);
    426     return;
    427   }
    428   
    429   // Choose axis to split on
    430   int axis;
    ...
    466   int median = (start+end) / 2;
    467   PhotonRecord** start_addr = &(photons[start]);
    ...
    484   switch( axis ) {
    485   case 0:
    486     select<PhotonRecord*, 0>( start_addr, 0, end-start-1, median-start );
    /// 
    ///                                  list     left   right       k
    ///                                                           
    ///                        list[left]..list[k-1] < list[k] < list[k+1]..list[right]
    ///
    ///            picks axis and spatial split value for median photon
    ///            partially orders photons PhotonRecord pointers 
    ///            such that they are above and below median value
    /// 
    ///            does a recursive tree split about the median, 
    ///
    ///
    ///    select.h, 
    ///        similar to Select1 algorithm from: Computer Algorithms C++ by Horowitz,...  p164 p154
    ///         
    ///
    487     photons[median]->axis = PPM_X;
    488     break;
    ...
    519   kd_tree[current_root] = *(photons[median]);
    ///
    ///       write median PhotonRecord into kd_tree   
    ///
    520   buildKDTree( photons, start, median, depth+1, kd_tree, 2*current_root+1, split_choice, bbmin,  leftMax );
    521   buildKDTree( photons, median+1, end, depth+1, kd_tree, 2*current_root+2, split_choice, rightMin, bbmax );
    ///
    ///           recurse on down doing left/right splits covering fewer and fewer photons until down to 0 or 1 
    ///           as the tree is perfectly balanced do not need pointers for its structure just regular indices
    ///
    ///                   current_root                        0   1   2   3   4   ...
    ///                            left: 2*current_root+1      1   3   5   7   9
    ///                           right: 2*current_root+2      2   4   6   8  10   
    ///
    ///                             

    522 }



select.h::

    127 /*
    128   returns the kth largest value in the list.  A side effect is that
    129   list[left]..list[k-1] < list[k] < list[k+1]..list[right].
    130 */
    131 
    132 template<class Elem, int axis> Elem select(Elem* list, int left, int right, int k)
    133 {


Using the map ppm_gather.cu::

    083 #define MAX_DEPTH 20 // one MILLION photons
     84 RT_PROGRAM void gather()
     85 {
     .. 
     87   PackedHitRecord rec = rtpass_output_buffer[launch_index];
     88   float3 rec_position = make_float3( rec.a.x, rec.a.y, rec.a.z );
     89   float3 rec_normal   = make_float3( rec.a.w, rec.b.x, rec.b.y );
     90   float3 rec_atten_Kd = make_float3( rec.b.z, rec.b.w, rec.c.x );
     91   uint   rec_flags    = __float_as_int( rec.c.y );
     92   float  rec_radius2  = rec.c.z;
     93   float  rec_photon_count = rec.c.w;
    ///
    ///
    ///    using photon map to find photons near to the rec  
    ///    without having to loop over all photons
    ///
    ///    starting at node 0, corresponding to median photon
    ///
    ...
    103   unsigned int stack[MAX_DEPTH];
    104   unsigned int stack_current = 0;
    105   unsigned int node = 0; // 0 is the start
    106 
    107 #define push_node(N) stack[stack_current++] = (N)
    108 #define pop_node()   stack[--stack_current]
    109 
    ///   explicit stack recursion
    ///
    110   push_node( 0 );
    ...
    114   uint num_new_photons = 0u;
    115   float3 flux_M = make_float3( 0.0f, 0.0f, 0.0f );
    116   uint loop_iter = 0;
    117   do {
    118 
    120     PackedPhotonRecord& photon = photon_map[ node ];
    121 
    122     uint axis = __float_as_int( photon.d.x );
    123     if( !( axis & PPM_NULL ) ) {
    124 
    125       float3 photon_position = make_float3( photon.a );
    126       float3 diff = rec_position - photon_position;
    127       float distance2 = dot(diff, diff);
    128 
    129       if (distance2 <= rec_radius2) {
    130         accumulatePhoton(photon, rec_normal, rec_atten_Kd, num_new_photons, flux_M);
    ///
    ///         only photons with dot(photon_normal, rec_normal) > 0.01 are accumulated
    ///         ie only photons incident on same side of surface are counted 
    ///         hmm do not need high precision photon_normal for this usage 
    ///
    ///         photon map stores incoming flux 
    ///
    131       }
    132 
    133       // Recurse
    134       if( !( axis & PPM_LEAF ) ) {
    135         float d;
    136         if      ( axis & PPM_X ) d = diff.x;
    137         else if ( axis & PPM_Y ) d = diff.y;
    138         else                      d = diff.z;
    139 
    140         // Calculate the next child selector. 0 is left, 1 is right.
    141         int selector = d < 0.0f ? 0 : 1;
    ///
    ///                    selector   (1 + selector)   (2 - selector)
    ///                        0         1                2
    ///                        1         2                1 
    ///            
    142         if( d*d < rec_radius2 ) {
    144                push_node( (node<<1) + 2 - selector );
    ///
    ///               when in viscinity check both left and right sides 
    ///
    145         }
    146 
    148         node = (node<<1) + 1 + selector;
    ///
    ///               continue left/right descent 
    ///            
    149       } else {
    150         node = pop_node();
    151       }
    152     } else {
    153       node = pop_node();
    154     }
    155     loop_iter++;
    156   } while ( node );

    158   // Compute new N,R
    159   float R2 = rec_radius2;
    160   float N = rec_photon_count;
    161   float M = static_cast<float>( num_new_photons ) ;
    162   float new_N = N + alpha*M;
    163   rec.c.w = new_N;  // set rec.photon_count
    164 
    165   float reduction_factor2 = 1.0f;
    166   float new_R2 = R2;
    167   if( M != 0 ) {
    168     reduction_factor2 = ( N + alpha*M ) / ( N + M );
    169     new_R2 = R2*( reduction_factor2 );
    170     rec.c.z = new_R2; // set rec.radius2
    171   }
    ...
    220   rtpass_output_buffer[launch_index] = rec;
    ///
    ///   for progressive accumulation to converge on something 
    ///   sensible need to write out adjusted radius for the next accumulation  
    ///
    221   float3 final_color = direct_flux + indirect_flux + ambient_light*rec_atten_Kd;
    222   output_buffer[launch_index] = make_float4(final_color);





PhotonMap/kdTree examples
-------------------------- 

* :google:`Pedersen progressive photon map`

  * see p20 of Pedersen Thesis for explanation of progressive photon mapping 

OppositeRenderer : Stian Pedersen Thesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.diva-portal.org/smash/get/diva2:655629/FULLTEXT01.pdf  
* http://apartridge.github.io/OppositeRenderer/master/masteroppgave.pdf 

* http://apartridge.github.io/OppositeRenderer/


Student Project : Thrust based kdtree construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
* http://xingdugpu.blogspot.tw
* http://xingdu.blogspot.tw/2012/05/gpu-path-tracer.html
* https://github.com/duxing/GPUFinal
* https://github.com/duxing/GPUFinal/blob/master/cuda_PhotonMapping/cuda_PhotonMapping/GPU_KDTree.h 

CUDA/Thrust kdTree part of FLANN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/mariusmuja/flann/tree/master/src/cpp/flann/algorithms
* https://github.com/mariusmuja/flann/blob/master/src/cpp/flann/algorithms/kdtree_cuda_builder.h



OptiX OpenGL Compositing
-------------------------

* https://github.com/nvpro-samples/gl_optix_composite
* http://rickarkin.blogspot.tw/2012/03/optix-is-ray-tracing-framework-it-can.html

* to allow depth correct 3D compositing must calulate the OptiX clipDepth.
  Raytracer alone does not need this, but to compostite with OpenGL lines etc.. it 
  is necessary

::

    // http://rickarkin.blogspot.tw/2012/03/optix-is-ray-tracing-framework-it-can.html
    // eyeDist:  distance from eye to the intersection point.
    // n:           near clipping plane
    // f:            far clipping plane
    __device__ float computeClipDepth( float eyeDist, float n, float f )
    {
        float clipDepth = (f+n)/(f-n) - (1/eyeDist)*2.0f*f*n/(f-n);
        clipDepth = clipDepth*0.5 + 0.5f;
        return clipDepth;
    }

* http://www.songho.ca/opengl/gl_projectionmatrix.html

* https://open.gl/depthstencils

The default clear value for the depth is 1.0f, which is equal to the depth of
your far clipping plane and thus the furthest depth that can be represented.
All potentially visible fragments will be closer than that.


* depth buffer contains values between 0.f to 1.f, 0.f/1.f corresponds to the closest/furthest distance

Perspective Projection Matrix::

    |  n/r   0     0              0          |
    |   0    n/t   0              0          |
    |   0    0    -(f+n)/(f-n)   -2fn/(f-n)  |
    |   0    0    -1              0          |

    |  n/r   0     0              0          |
    |   0    n/t   0              0          |
    |   0    0     A              B          |
    |   0    0    -1              0          |


    [ 0,0, z, 1]   = [ 0, 0, Az+B, -z ]


    Perspective divide (division by w=-1) is done by the pipeline 
    yielding Z value:


    Az + B
    -------  =  -A - B/z   = (f+n)/(f-n) + 2*fn/(f-n)/dist
      -z
   
                             (f+n) + 2*fn/dist
                            ----------------------
                                 (f-n)

                   dist = -f

                             f - n
                            -------  = 1.
                             f - n 

                  dist = -n
 
                             (f+n) - 2f
                           ------------- = -1.
                              f - n  


    n and f are +ve and f > n  
    dist must be negative for proper NDC (ie in range -1:1) 


* http://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer

::

    varying float depth; // Linear depth, in world units
    void main(void)
    {
        float A = gl_ProjectionMatrix[2].z;
        float B = gl_ProjectionMatrix[3].z;
        gl_FragDepth  = 0.5*(-A*depth + B) / depth + 0.5;
    }






The second problem is to use the generated depth buffer of Optix into OpenGL.
Actually it is totally OpenGL operations. But maybe its not a daily used
process like draw a triangle or shading a scene object, so there is little
resource could be found on the web.  My realization of the depth value
construction is  also attached as below, where depthImg contains per pixel
depth value, coloredImg contains per pixel color value.

Hmm this is old OpenGL::

    // http://rickarkin.blogspot.tw/2012/03/optix-is-ray-tracing-framework-it-can.html

    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);
    // these store the modes before dillying with them

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    //
    // sets pixel storage modes that affect the operation of subsequent glReadPixels 
    // as well as the unpacking of texture patterns (see glTexImage2D and glTexSubImage2D)
    //
    // GL_UNPACK_ALIGNMENT
    // Specifies the alignment requirements for the start of each pixel row in memory. 
    // The allowable values are 
    // 1 (byte-alignment), 
    // 2 (rows aligned to even-numbered bytes), 
    // 4 (word-alignment)
    // 8 (rows start on double-word boundaries).

    // draw coloredImg pixels, skipping the depth 
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_FALSE);    
    glWindowPos2i(0, 0);
    glDrawPixels(w, h, GL_RGBA , GL_FLOAT, coloredImg);

    // draw depthImg pixels, just the depth 
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_ALWAYS);
    glWindowPos2i(0, 0);
    glDrawPixels(w, h, GL_DEPTH_COMPONENT , GL_FLOAT, depthImg);

    // regain original stored state 
    glPopClientAttrib();
    glPopAttrib(); // GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT


Whats the modern equivalent ?

* https://open.gl/depthstencils


Where to composite ?
----------------------

When compositing a pixel level mixture OpenGL render and OptiX texture is needed with depth 
buffer control of which is frontmost.
Currently either Scene::render or ORenderer::render get invoked.

ggv-/main.cc::

    1083 void App::render()
    1084 {
    1085     if(m_opticks->isCompute()) return ;
    1086 
    1087     m_frame->viewport();
    1088     m_frame->clear();
    1089 
    1090 #ifdef OPTIX
    1091     if(m_interactor->getOptiXMode()>0 && m_otracer && m_orenderer)
    1092     {
    1093         unsigned int scale = m_interactor->getOptiXResolutionScale() ;
    1094         m_otracer->setResolutionScale(scale) ;
    1095         m_otracer->trace();
    1096         //LOG(debug) << m_ogeo->description("App::render ogeo");
    1097 
    1098         m_orenderer->render();
    1099     }
    1100     else
    1101 #endif
    1102     {
    1103         m_scene->render();
    1104     }


OptiX geometry rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply presents the OptiX derived texture 

optixrap-/ORenderer::render invokes 

* optixrap-/OFrame::push_PBO_to_Texture(m_texture_id)
* oglrap-/Renderer::render "tex" pipeline 

* gl/tex/vert.glsl just gets called on the 4 corners of the quad texture  

gl/tex/frag.glsl just samples the texture::

     04 in vec3 colour;
      5 in vec2 texcoord;
      6 
      7 out vec4 frag_colour;
      8 uniform sampler2D texSampler;
      9 
     10 void main () 
     11 {
     12    frag_colour = texture(texSampler, texcoord);


OpenGL scene rendering
~~~~~~~~~~~~~~~~~~~~~~~

Implemented in oglrap-/Scene.cc with multiple (order ~10) renderers
for the constituents of the scene like 

* nrm/nrmvec: global geometry (triangulated)
* inrm: instanced geometry (triangulated)
* axis: axis
* p2l: genstep
* pos: photons
* rec/altrec/devrec: records 


Composited rendering
~~~~~~~~~~~~~~~~~~~~~~

Makes most sense to add depth buffered OptiX geometry as another
constituent of the OpenGL scene rendering, with one more renderer.



Peek at OptiX internals via stack trace
----------------------------------------

::

    OBuffer::mapGLToOptiX (createBufferFromGLBO) 1  size 30
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Memory allocation failed (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: driver().cuGraphicsGLRegisterBuffer(&cudaResource, id, flags) returned (2): Out of memory, [3735964])
    Abort trap: 6


OptiX Multi-GPU debug
----------------------

* https://devtalk.nvidia.com/default/topic/853345/?comment=4597946


Issue : reboot required to recover from timeout exceptions 
-----------------------------------------------------------

When things go wrong in an OptiX program (eg an unintended infinite loop)
the usual result is a Timeout exception. 
Unfortunately on Macbook Pro 2013 with Geforce 750M,
with CUDA 7.0, OptiX 3.8 and driver ??? following the report of the exception
the machine freezes requiring a hardware reboot to recover.

Perhaps some cleanup code that runs when Timeout exceptions occur  
could prevent the need for rebooting ?

Maybe try to capture what is going on the next time this happens using 
OPTIX_API_CAPTURE=1 and send trace to optix-help@nvidia.com.
For instructions regarding traces see:

* https://devtalk.nvidia.com/default/topic/803116/optix/error-with-rtpmodelupdate/

TODO
~~~~

Try catching the exception and exiting immediately ?

* https://devtalk.nvidia.com/default/topic/734914/optix/optix-bug-crash-with-cuda-error-kernel-ret-700-when-not-rtprinting-anything-small-demo-code-/



OptiX Versions
-----------------

::

    3.5.1         02/21/2014 05:12:40    
    3.6.0         06/11/2014 13:32:05
    3.6.3         09/27/2014 18:52:56    CUDA R331 driver or later 
    3.7.0-beta2                                                   
    3.7.0-beta3   01/30/2015 17:35:59    CUDA R343 driver
    3.7.0         02/24/2015 22:53:11
    3.8.0-beta    03/14/2015 17:04:51
    3.8.0         06/01/2015 07:18:19    CUDA R346 driver or later, for Mac the driver extension module supplied with CUDA 7.0 will need to be installed. 


OptiX Download and Unpack 
-----------------------------

Download and mount dmg::

    optix-ftp    # web interface, click on .dmg 

    mv ~/Downloads/NVIDIA-OptiX-SDK-3.8.0-mac64.dmg $(optix-download-dir)/

    open NVIDIA-OptiX-SDK-3.8.0-mac64.dmg   # mounts volume containing NVIDIA-OptiX-SDK-3.8.0-mac64.pkg 

Examine pkg contents using lsbom, verify that target is all beneath /Developer/OptiX::

    simon:~ blyth$ lsbom /Volumes/NVIDIA-OptiX-SDK-3.8.0-mac64/NVIDIA-OptiX-SDK-3.8.0-mac64.pkg/Contents/Archive.bom
    .   40755   501/0
    ./Developer 40755   501/0
    ./Developer/OptiX   40755   0/80
    ./Developer/OptiX/SDK   40755   0/80
    ./Developer/OptiX/SDK/CMake 40755   0/80
    ./Developer/OptiX/SDK/CMake/CompilerInfo.cmake  100644  0/80    3392    2636181668
    ./Developer/OptiX/SDK/CMake/ConfigCompilerFlags.cmake   100644  0/80    15064   2806959999
    ./Developer/OptiX/SDK/CMake/CopyDLL.cmake   100644  0/80    2850    1542474852
    ./Developer/OptiX/SDK/CMake/FindCUDA    40755   0/80
    ./Developer/OptiX/SDK/CMake/FindCUDA/make2cmake.cmake   100644  0/80    3532    494331911
    ./Developer/OptiX/SDK/CMake/FindCUDA/parse_cubin.cmake  100644  0/80    3666    547407452
    ./Developer/OptiX/SDK/CMake/FindCUDA/run_nvcc.cmake 100644  0/80    13360   3696680251
    ./Developer/OptiX/SDK/CMake/FindCUDA.cmake  100644  0/80    74607   1732037123
    ./Developer/OptiX/SDK/CMake/FindDX.cmake    100644  0/80    1414    2129523030
    ./Developer/OptiX/SDK/CMake/FindOptiX.cmake 100644  0/80    6408    1762934238
    ./Developer/OptiX/SDK/CMake/FindSUtilGLUT.cmake 100644  0/80    2747    312119594 
    ...

Prior to unpacking delete my symbolic link::

    simon:~ blyth$ l /Developer/
    total 8
    drwxr-xr-x  4 root  wheel  136 Jun 29 17:05 NVIDIA
    lrwxr-xr-x  1 root  wheel   11 Feb  2 12:53 OptiX -> OptiX_370b2
    drwxr-xr-x  7 root  admin  238 Jan 22 16:17 OptiX_301
    drwxr-xr-x  7 root  admin  238 Dec 18  2014 OptiX_370b2

    simon:~ blyth$ sudo rm /Developer/OptiX

Open .pkg from mounted .dmg, run the GUI installer, 
then rename and symbolicate::

    simon:Developer blyth$ sudo mv OptiX OptiX_380
    simon:Developer blyth$ sudo ln -s OptiX_380 OptiX 
    simon:Developer blyth$ l
    total 8
    lrwxr-xr-x  1 root  wheel    9 Jun 29 20:48 OptiX -> OptiX_380
    drwxr-xr-x  4 root  wheel  136 Jun 29 17:05 NVIDIA
    drwxr-xr-x  7 root  admin  238 May 29 00:23 OptiX_380
    drwxr-xr-x  7 root  admin  238 Jan 22 16:17 OptiX_301
    drwxr-xr-x  7 root  admin  238 Dec 18  2014 OptiX_370b2


Try Precompiled Samples
--------------------------

::

    cd /Developer/OptiX/SDK-precompiled-samples

    simon:SDK-precompiled-samples blyth$ open sample1.app
    simon:SDK-precompiled-samples blyth$ open sample2.app
    simon:SDK-precompiled-samples blyth$ open cook.app
    simon:SDK-precompiled-samples blyth$ open path_tracer.app
    simon:SDK-precompiled-samples blyth$ open instancing.app
    simon:SDK-precompiled-samples blyth$ open tutorial.app
    simon:SDK-precompiled-samples blyth$ open sphereTessellate.app
    simon:SDK-precompiled-samples blyth$ open whitted.app


Test can compile samples
--------------------------

::

    simon:SDK-precompiled-samples blyth$ optix-samples-get-all
    optix-samples-get-all copy all samples to somewhere writable
    simon:SDK-precompiled-samples blyth$ 



OptiX 3.8,  05/30/2015
------------------------

https://devtalk.nvidia.com/default/topic/836902/optix/optix-3-8-final-release-is-out-/

What is this Mac Driver Extension Module ? 
--------------------------------------------

Presumably CUDA.kext::

    simon:cuda blyth$ file  /System/Library/Extensions/CUDA.kext/Contents/MacOS/CUDA 
    /System/Library/Extensions/CUDA.kext/Contents/MacOS/CUDA: Mach-O universal binary with 1 architecture
    /System/Library/Extensions/CUDA.kext/Contents/MacOS/CUDA (for architecture x86_64): Mach-O 64-bit kext bundle x86_64

The CUDA uninstalll does::

    kextunload /System/Library/Extensions/CUDA.kext

::

    simon:cuda blyth$ kextstat | head -1 && kextstat | grep nvidia
    Index Refs Address            Size       Wired      Name (Version) <Linked Against>
      107    2 0xffffff7f80c52000 0x274000   0x274000   com.apple.nvidia.driver.NVDAResman (8.2.6) <83 74 71 11 5 4 3 1>
      108    0 0xffffff7f80ed1000 0x1ad000   0x1ad000   com.apple.nvidia.driver.NVDAGK100Hal (8.2.6) <107 11 4 3>
      127    0 0xffffff7f81dbe000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) <4 1>


Seems no Mac equivalent of the R346 R355 R343 ...
---------------------------------------------------

Kexts have version numbers (above 1.1.0), 
but they are not referred to, instead just get the text::

    For the Mac, the driver extension module supplied with CUDA 7.0 
    will need to be installed.


Release Notes OptiX 3.8.0 (May 2015)
------------------------------------------

The CUDA R346 or later driver is required. For the Mac, 
the driver extension module supplied with CUDA 7.0 will need to be installed.

Driver R355 or newer is required for optimal performance on Maxwell GPUs. 
OptiX 3.8 supports Maxwell GPUs on earlier drivers, but at up to 20% lower performance.

A CUDA compiler bug that causes timeouts or crashes on Maxwell cards has been
worked around. Unfortunately, this bug fix causes a slowdown of up to 20% on
Maxwell cards unless driver R355 or newer is used.

CMake 2.8.12 http://www.cmake.org/cmake/resources/software.html


Release Notes OptiX Version 3.7 beta 3 (January 2015)
--------------------------------------------------------

The CUDA R343 or later driver is required. The latest available WHQL drivers
are highly recommended. For the Mac, the driver extension module supplied with
CUDA 5.0 or later will need to be installed. Note that the Linux and Mac
drivers can only be obtained from the CUDA 6.5 download page at the moment.

SLI is not required for OptiX to use multiple GPUs, and it interferes when
OptiX uses either D3D or OpenGL interop. Disabling SLI will not degrade OptiX
performance and will provide a more stable environment for OptiX applications
to run. SLI is termed "Multi-GPU mode" in recent NVIDIA Control Panels, with
the correct option being "Disable multi-GPU mode" to ensure OptiX is not
encumbered by graphics overhead.

Release Notes OptiX Version 3.6.3 (September 2014)
----------------------------------------------------

The CUDA R331 or later driver is required. The latest available WHQL drivers
are highly recommended (343 or later for Windows, 343 for Linux and the CUDA
6.0 driver extension for Mac). For the Mac, the driver extension module
supplied with CUDA 5.0 or later will need to be installed. Note that the Linux
and Mac drivers can only be obtained from the CUDA 6.0 download page at the
moment.

Release Notes OptiX Version 3.5 (January 2013)
----------------------------------------------------

The CUDA R319 or later driver is required. The latest available WHQL drivers
are highly recommended (320.92 or later for Windows, 319.60 for Linux and the
CUDA 5.5 driver extension for Mac). For the Mac, the driver extension module
supplied with CUDA 5.0 or later will need to be installed. Note that the Linux
and Mac drivers can only be obtained from the CUDA 5.5 download page at the
moment.


Update OptiX version and build samples
---------------------------------------

::

    -bash-4.1$ optix-linux-jump 370         # modify the symbolic link
    OptiX -> NVIDIA-OptiX-SDK-3.7.0-linux64

    -bash-4.1$ optix-name
    NVIDIA-OptiX-SDK-3.7.0-linux64     

    -bash-4.1$ optix-samples-get-all   ## copy samples, to avoid touching originals

    -bash-4.1$ optix-samples-cmake     ## fails due to cmake version 

    -bash-4.1$ optix-samples-cmake-kludge   ## kludge the requirement, seems to work with 2.6.4
    cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
    cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

    -bash-4.1$ optix-samples-cmake     ## now completes
    -bash-4.1$ optix-samples-make      ## 


Determine Driver Version on Linux (on Mac use SysPref panel)
----------------------------------------------------------------

With nvidia-smi or::

    -bash-4.1$ cat /proc/driver/nvidia/version   # original old driver
    NVRM version: NVIDIA UNIX x86_64 Kernel Module  319.37  Wed Jul  3 17:08:50 PDT 2013
    GCC version:  gcc version 4.4.7 20120313 (Red Hat 4.4.7-4) (GCC) 

    -bash-4.1$ cat /proc/driver/nvidia/version   # updated Feb ~7 2015 
    NVRM version: NVIDIA UNIX x86_64 Kernel Module  340.65  Tue Dec  2 09:50:34 PST 2014
    GCC version:  gcc version 4.4.7 20120313 (Red Hat 4.4.7-4) (GCC) 
    -bash-4.1$ 

hgpu01 install
--------------

Manual CUDA::

    -bash-4.1$ cat /etc/redhat-release 
    Scientific Linux release 6.5 (Carbon)

    -bash-4.1$ rpm -qf /usr/local/cuda-5.5
    file /usr/local/cuda-5.5 is not owned by any package

Hmm getting "OptiX Error: Invalid context", with 370b3 
I was getting this on laptop until I updated the driver.

::

    scp /Users/blyth/Downloads/NVIDIA-OptiX-SDK-3.5.1-linux64.run L6:/dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn/
    scp /Users/blyth/Downloads/NVIDIA-OptiX-SDK-3.6.3-linux64.run L6:/dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn/
    scp /Users/blyth/Downloads/NVIDIA-OptiX-SDK-3.7.0-linux64.sh L6:/dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn/

    -bash-4.1$ chmod ugo+x NVIDIA-OptiX-SDK-3.5.1-linux64.run 
    -bash-4.1$ chmod ugo+x NVIDIA-OptiX-SDK-3.6.3-linux64.run 
    -bash-4.1$ chmod ugo+x NVIDIA-OptiX-SDK-3.7.0-linux64.sh 

    -bash-4.1$ ./NVIDIA-OptiX-SDK-3.5.1-linux64.run --prefix=. --include-subdir
    -bash-4.1$ ./NVIDIA-OptiX-SDK-3.6.3-linux64.run --prefix=. --include-subdir
    -bash-4.1$ ./NVIDIA-OptiX-SDK-3.7.0-linux64.sh --prefix=. --include-subdir

    -bash-4.1$ ln -sfnv NVIDIA-OptiX-SDK-3.5.1-PRO-linux64 OptiX
    -bash-4.1$ ln -sfnv NVIDIA-OptiX-SDK-3.6.3-linux64 OptiX
    -bash-4.1$ ln -sfnv NVIDIA-OptiX-SDK-3.7.0-linux64 OptiX

    -bash-4.1$ pwd
    /dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn/OptiX/SDK-precompiled-samples

    -bash-4.1$ LD_LIBRARY_PATH=. ./sample3
    OptiX 3.7.0
    Number of Devices = 2
    ...
    Constructing a context...
    OptiX Error: Invalid context
    (/root/sw/wsapps/raytracing/rtsdk/rel3.7/samples/sample3/sample3.c:96)

    -bash-4.1$ LD_LIBRARY_PATH=. ./sample3
    OptiX 3.6.3
    Number of Devices = 2
    ...
    Constructing a context...
      Created with 2 device(s)
      Supports 2147483647 simultaneous textures
      Free memory:
        Device 0: 4952547328 bytes
        Device 1: 4952547328 bytes

    -bash-4.1$ LD_LIBRARY_PATH=. ./sample3
    OptiX 3.5.1
    Number of Devices = 2
    ...
    Constructing a context...
      Created with 2 device(s)
      Supports 2147483647 simultaneous textures
      Free memory:
        Device 0: 4952547328 bytes
        Device 1: 4952547328 bytes

::

    -bash-4.1$ LD_LIBRARY_PATH=. ./sample7 -f sample7.ppm
    OptiX Error: Invalid context (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Unable to set the CUDA device., [3735714])
    -bash-4.1$ 


Partial override warning
---------------------------

::

    /Developer/OptiX/include/optixu/optixpp_namespace.h(588): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"

* http://stackoverflow.com/questions/21462908/warning-overloaded-virtual-function-baseprocess-is-only-partially-overridde


OptiX Transform (Programming Guide)
----------------------------------------

A transform node is used to represent a projective transformation of its
underlying scene geometry. The transform must be assigned exactly one child of
type rtGroup, rtGeometryGroup, rtTransform, or rtSelector, using
rtTransformSetChild. That is, the nodes below a transform may simply be
geometry in the form of a geometry group, or a whole new subgraph of the scene.

The transformation itself is specified by passing a 4×4 floating point matrix
(specified as a 16-element one-dimensional array) to rtTransformSetMatrix.
Conceptually, it can be seen as if the matrix were applied to all the
underlying geometry. However, the effect is instead achieved by transforming
the rays themselves during traversal. This means that **OptiX does not rebuild
any acceleration structures when the transform changes**.

Note that the transform child node may be shared with other graph nodes. That
is, a child node of a transform may be a child of another node at the same
time. This is often useful for instancing geometry.


::

     transform
         geometry_group
               


::

    delta:bin blyth$ optix-;optix-samples-cppfind Transform -l
    /usr/local/env/cuda/OptiX_380_sdk/glass/glass.cpp
    /usr/local/env/cuda/OptiX_380_sdk/hybridShadows/hybridShadows.cpp
    /usr/local/env/cuda/OptiX_380_sdk/instancing/instancing.cpp
    /usr/local/env/cuda/OptiX_380_sdk/isgReflections/isgReflections.cpp
    /usr/local/env/cuda/OptiX_380_sdk/isgShadows/isgShadows.cpp
    /usr/local/env/cuda/OptiX_380_sdk/primeInstancing/primeInstancing.cpp
    /usr/local/env/cuda/OptiX_380_sdk/sample7/sample7.cpp
    /usr/local/env/cuda/OptiX_380_sdk/sutil/OptiXMesh.cpp
    /usr/local/env/cuda/OptiX_380_sdk/sutil/OptiXMeshImpl.cpp
    /usr/local/env/cuda/OptiX_380_sdk/swimmingShark/fishMonger.cpp

Usage
~~~~~~~~

* wrapping moving pieces of geometry into Transforms allows position
  to be changed without rebuilding acceleration structures 


Thoughts on applying *Transform* instancing to complex/large geometries
-----------------------------------------------------------------------------

* Current geocache machinery is flat using final transforms applied to every volume.
  This works fine when treating everything as triangles and yields a very simple
  *convertDrawable*  

  * optix::Geometry GMergedMeshOptiXGeometry::convertDrawable(GMergedMesh* mergedmesh)

* Attempting to operate as if every solid making up the PMT is independant 
  each with a global transform (as the flat geocache encourages) 
  would yield an unnecessarily complicated OptiX geometry of overlapping transforms 

  * this would likely not work and even if it did would be fragile and difficult to 
    move, consider for example a analogous treatment of movable calibration sources 

* PMTs (and calibration sources) are not simple single volumes, 
  they are a collection of volumes arranged via transforms wrt to each other 
  with the assembly placed in the wider detector via a "placement" transform

* can sub-assemblies be auto-identified by finding repeated transform/meshIndex sub-trees ?
 
  * construct transform/meshIndex digests for the tree beneath every node
  * look for repeated such digests and locate the parent placement transforms 

* need to create local assembly frame vertices 


OptiX instancing in formum
-----------------------------

* https://devtalk.nvidia.com/default/topic/647610/optix/instancing-for-geometry/




Instancing Example
-------------------

::

    214   // Set up instances
    215   Group group = m_context->createGroup();
    216   group->setChildCount(m_num_instances);
    217   optix::Aabb aabb = loader.getSceneBBox();
    218 
    219   unsigned int dimension = static_cast <unsigned int> ( ceilf( powf( static_cast <float> (m_num_instances), .3333333f )));
    220   unsigned int dimension2 = dimension*dimension;
    221   optix::Matrix4x4 m;
    222   for(unsigned int i = 0; i < m_num_instances; ++i) {
    223     Transform xform = m_context->createTransform();
    224     xform->setChild(m_geometry_group);
    225     group->setChild(i, xform);
    226 
    227     if (m_grid_mode) {
    228       float tx = static_cast <float>( (i%dimension));
    229       float ty = static_cast <float>( ((i%dimension2)/dimension));
    230       float tz = static_cast <float>( (i/dimension2) );
    231 
    232       m = optix::Matrix4x4::translate(make_float3( tx, ty, tz ) * (- m_grid_distance));
    233     } else {
    234       float tx = 4 * aabb.extent(0) * (randFloat()*2.f-1.f);
    235       float ty = 4 * aabb.extent(1) * (randFloat()*2.f-1.f);
    236       float tz = 4 * aabb.extent(2) * (randFloat()*2.f-1.f);
    237       float3 randomish_dir = normalize( make_float3( randFloat()*2.f-1.f, randFloat()*2.f-1.f, randFloat()*2.f-1.f ) );
    238 
    239       optix::Matrix4x4 trans = optix::Matrix4x4::translate(make_float3( tx, ty, tz ) );
    240       optix::Matrix4x4 rot   = optix::Matrix4x4::rotate(randFloat()*2.f*M_PIf, randomish_dir);
    241 
    242       m = rot * trans;
    243     }
    244 
    245     xform->setMatrix(false, m.getData(), 0);
    246   }
    ///
    ///  
    ///    instances "assembly" comprised of a 
    ///    group of N xform, each holding a single repeated m_geometry_group
    ///         
    ///          group
    ///             xform_0 < m_geometry_group
    ///             xform_1 < m_geometry_group
    ///             xform_2 < m_geometry_group  
    ///             ...
    ///             xform_N < m_geometry_group
    ///
    ///
    ///    hmm how about geometry instances and adding material ?
    ///
    247 
    248   Acceleration top_level_bvh = m_context->createAcceleration("Bvh", "Bvh");
    249   group->setAcceleration(top_level_bvh);
    250 
    251   m_context[ "top_object" ]->set( group );
    252   m_context[ "top_shadower" ]->set( group );





How to persist tree of transforms in the geocache ?
-----------------------------------------------------


OptiX Instancing : 20k teapots
---------------------------------

::

    optix-;optix-samples-cd bin
    ./instancing -i 20000 -n 
    ./instancing -i 20000 --grid=100x100x100



OptiX Selector (Programming Guide)
-------------------------------------

A selector is similar to a group in that it is a collection of higher level
graph nodes. The number of nodes in the collection is set by
rtSelectorSetChildCount, and the individual children are assigned with
rtSelectorSetChild. Valid child types are rtGroup, rtGeometryGroup,
rtTransform, and rtSelector.

The main difference between selectors and groups is that selectors do not have
an acceleration structure associated with them. Instead, a visit program is
specified with rtSelectorSetVisitProgram. This program is executed every time a
ray encounters the selector node during graph traversal. The program specifies
which children the ray should continue traversal through by calling
*rtIntersectChild*.

A typical use case for a selector is dynamic (i.e. per-ray) level of detail: an
object in the scene may be represented by a number of geometry nodes, each
containing a different level of detail version of the object. The geometry
groups containing these different representations can be assigned as children
of a selector. The visit program can select which child to intersect using any
criterion (e.g. based on the footprint or length of the current ray), and
ignore the others.

As for groups and other graph nodes, child nodes of a selector can be shared
with other graph nodes to allow flexible instancing.


Selector Examples
~~~~~~~~~~~~~~~~~~

::

    delta:ggeo blyth$ optix-;optix-samples-cufind IntersectChild
    /usr/local/env/cuda/OptiX_380_sdk/device_exceptions/device_exceptions.cu:    rtIntersectChild( 1 );
    /usr/local/env/cuda/OptiX_380_sdk/device_exceptions/device_exceptions.cu:    rtIntersectChild( 0 );
    /usr/local/env/cuda/OptiX_380_sdk/sample8/selector_example.cu:  rtIntersectChild( index );
    delta:ggeo blyth$ 

    optix-;optix-samples-cd bin

    delta:bin blyth$ ./sample8 -h


OptiX glass
------------

* https://devtalk.nvidia.com/default/topic/458979/?comment=3263252

Overlapping geometry problem


OptiX 3.6.3 problems
----------------------


* https://devtalk.nvidia.com/default/topic/763478/simplest-optix-code-unable-to-set-cuda-device/

::

    -bash-4.1$ LD_LIBRARY_PATH=. ./sample7 -f sample7.ppm
    OptiX Error: Invalid context (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Unable to set the CUDA device., [3735714])
    -bash-4.1$ 
 

OptiX_370b2 rtPrintf bizarre bug
----------------------------------

Whilst debugging wavelength texture lookups with
a program which is only invoked for the single touched pixel 
under the mouse::

     33 RT_PROGRAM void closest_hit_touch()
     34 {
     35   prd_touch.result = contrast_color ;
     36   prd_touch.node = node_index ; 
     37   
     38   prd_touch.texlookup_b = wlookup( NM_BLUE  , 0.5f ) ;
     39   prd_touch.texlookup_g = wlookup( NM_GREEN , 0.5f ) ;
     40   prd_touch.texlookup_r = wlookup( NM_RED   , 0.5f ) ;
     41   
     42   for(int i=-5 ; i < 45 ; i++ )
     43   { 
     44      float wl = wavelength_domain.x + wavelength_domain.z*i ;
     45      float4 lookup = wlookup( wl, 0.5f ); 
     46      rtPrintf("material1.cu::closest_hit_touch node %d   i %2d wl %10.3f   lookup  %10.3f %10.3f %10.3f %10.3f \n",
     47         node_index,
     48         i,
     49         wl,
     50         lookup.x,
     51         lookup.y,
     52         lookup.z,
     53         lookup.w);
     54   }     
     55 }

Get the expected output, BUT on splitting the rtPrintf 
into two calls get **Unknown error**.::

    OptiX Error: Unknown error (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Assertion failed: "Traits::getNext(m_cur) != 0", [7143516]) 

Unfortunately the only way to discover the source of
the problem is by "binary search" trial and error. Moral:

* never make largescale changes to optix programs without testing, 
  always test after making small focussed changes 

* limit use rtPrintf to only one per program ?

* dont leave rtPrintf lying around, just use one at a time whilst debugging
  and then comment them out



Transparency/blending
-----------------------

* https://developer.nvidia.com/content/transparency-or-translucency-rendering

* http://casual-effects.blogspot.tw/2014/03/weighted-blended-order-independent.html



OptiX rtDeclareVariable attribute variables
---------------------------------------------


From optix-pdf section *4.1.4 Attribute Variables* p35

Attribute variables provide a mechanism for communicating data between the
intersection program and the shading programs (e.g., surface normal, texture
coordinates). Attribute variables may only be written in an intersection
program between calls to rtPotentialIntersection and rtReportIntersection.


Do I need multiple OptiX materials ?
--------------------------------------

Aiming for different materials to just corresponds to 
different substance indices used to lookup into a single texture, 
as such no need for separate OptiX materials ? So just need::

    rtReportIntersection(0)

Only Bialkali needs some different behaviour to do PMT id lookups, 
so that could benefit from being a different material.

In mesh_intersect can communicate the primIdx via attribute 
in order to do the substance lookup in closestHit 
Remember mesh_intersect gets called the most, so avoid
doing anything in there that can be done elsewhere.





OptiX and atomics
-------------------

* https://devtalk.nvidia.com/default/topic/522795/optix/atomic-buffer-operations/

  see zoneplate sample


  One thing to keep in mind is that atomic operations will not work in multi-GPU
  situations unless you specify RT_BUFFER_GPU_LOCAL. In that case the data stays
  resident on the device and can only be read by the device that wrote it.


OptiX and curand ?
-------------------

* :google:`optix curand`

* https://devtalk.nvidia.com/search/more/sitecommentsearch/curand%20optix/
* https://devtalk.nvidia.com/default/topic/759883/random-number-streams/?offset=1
* https://devtalk.nvidia.com/default/topic/770325/curand_init-within-optix/

Suggest that it can be made to work

Chroma curand
~~~~~~~~~~~~~~

chroma/chroma/cuda/random.h::

    001 #ifndef __RANDOM_H__
      2 #define __RANDOM_H__
      3 
      4 #include <curand_kernel.h>
      5 
      6 #include "physical_constants.h"
      7 #include "interpolate.h"
      8 
      9 __device__ float
     10 uniform(curandState *s, const float &low, const float &high)
     11 {
     12     return low + curand_uniform(s)*(high-low);
     13 }
    ///   all the random funcs have curandState* s argument 
    ...
    135 __global__ void
    136 init_rng(int nthreads, curandState *s, unsigned long long seed,
    137      unsigned long long offset)
    138 {
    139     int id = blockIdx.x*blockDim.x + threadIdx.x;
    140 
    141     if (id >= nthreads)
    142     return;
    143 
    144     curand_init(seed, id, offset, &s[id]);
    145 }


chroma/chroma/cuda/propagate_hit.cu::

    128 __global__ void
    129 propagate_hit(
    ...
    134       curandState *rng_states,
    ...   
    ...  
    164     int id = blockIdx.x*blockDim.x + threadIdx.x;
    165 
    166     if (id >= nthreads)
    167     return;
    168 
    169     g = &sg;
    170 
    171     curandState rng = rng_states[id];
    ...
    208             generate_cerenkov_photon(p, cs, rng );


chroma/chroma/gpu/tools.py::

    107 init_rng_src = """
    108 #include <curand_kernel.h>
    109 
    110 extern "C"
    111 {
    112 
    113 __global__ void init_rng(int nthreads, curandState *s, unsigned long long seed, unsigned long long offset)
    114 {
    115     int id = blockIdx.x*blockDim.x + threadIdx.x;
    116 
    117     if (id >= nthreads)
    118         return;
    119 
    120     curand_init(seed, id, offset, &s[id]);
    121 }
    122 
    123 } // extern "C"
    124 """
    125 
    126 def get_rng_states(size, seed=1):
    127     "Return `size` number of CUDA random number generator states."
    128     rng_states = cuda.mem_alloc(size*characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
    129 
    130     module = pycuda.compiler.SourceModule(init_rng_src, no_extern_c=True)
    131     init_rng = module.get_function('init_rng')
    132 
    133     init_rng(np.int32(size), rng_states, np.uint64(seed), np.uint64(0), block=(64,1,1), grid=(size//64+1,1))
    134 
    135     return rng_states



optix exception codes
-----------------------

/Developer/OptiX/include/internal/optix_declarations.h::

    197 typedef enum
    198 {
    199   RT_EXCEPTION_PROGRAM_ID_INVALID           = 0x3EE,    /*!< Program ID not valid       */
    200   RT_EXCEPTION_TEXTURE_ID_INVALID           = 0x3EF,    /*!< Texture ID not valid       */
    201   RT_EXCEPTION_BUFFER_ID_INVALID            = 0x3FA,    /*!< Buffer ID not valid        */
    202   RT_EXCEPTION_INDEX_OUT_OF_BOUNDS          = 0x3FB,    /*!< Index out of bounds        */
    203   RT_EXCEPTION_STACK_OVERFLOW               = 0x3FC,    /*!< Stack overflow             */
    204   RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS   = 0x3FD,    /*!< Buffer index out of bounds */
    205   RT_EXCEPTION_INVALID_RAY                  = 0x3FE,    /*!< Invalid ray                */
    206   RT_EXCEPTION_INTERNAL_ERROR               = 0x3FF,    /*!< Internal error             */
    207   RT_EXCEPTION_USER                         = 0x400,    /*!< User exception             */
    208 
    209   RT_EXCEPTION_ALL                          = 0x7FFFFFFF  /*!< All exceptions        */
    210 } RTexception;


testing optix with curand
---------------------------

When attempting to use curand subsequences getting RT_EXCEPTION_STACK_OVERFLOW::

    Caught exception 0x3FC at launch index (144,0)
    Caught exception 0x3FC at launch index (0,0)
    Caught exception 0x3FC at launch index (96,0)

with the MeshViewer original setting of stack size::

    m_context->setStackSize( 1180 );

Winding up the stack size to 10000 succeeds to run, but unusably slowly.

things to try
~~~~~~~~~~~~~~

* hmm, could subsequent optix launches reuse a buffer initialized on the
  first launch ?  to enable a single initializing launch 
* hmm, but optix is unusable with the large stacksizes needed 
  for curand_init with subsequences and probably changing stack size 
  will invalidate the context ?
* what about using plain CUDA kernel call to 
  do curand_init and prepare the curandState buffer for interop
  with subsequent optix launches 



OptiX and CUDA interop
------------------------

Doc says, if the application creates a CUDA context before OptiX, 
the applicaton should make sure to use the below
to ensure subsequent maximum performance from OptiX.::

    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax);


::

    delta:OptiX_370b2_sdk blyth$ find . -name '*.cpp' -exec grep -H cudaSet {} \;
    ./ocean/ocean.cpp:    cudaSetDevice(m_cuda_device);
    ./ocean/ocean.cpp:  cudaSetDevice( m_cuda_device );
    ./simplePrime/simplePrimeCommon.cpp:    CHK_CUDA( cudaSetDevice(i) );
    ./simplePrimeInstancing/simplePrimeCommon.cpp:    CHK_CUDA( cudaSetDevice(i) );
    ./simplePrimeMasking/simplePrimeCommon.cpp:    CHK_CUDA( cudaSetDevice(i) );
    ./simplePrimepp/simplePrimeCommon.cpp:    CHK_CUDA( cudaSetDevice(i) );
    ./simplePrimeppMultiBuffering/simplePrimeCommon.cpp:    CHK_CUDA( cudaSetDevice(i) );
    ./simplePrimeppMultiGpu/simplePrimeCommon.cpp:    CHK_CUDA( cudaSetDevice(i) );
    ./simplePrimeppMultiGpu/simplePrimeppMultiGpu.cpp:      CHK_CUDA( cudaSetDevice(int(i)) );
    ./simplePrimeppMultiGpu/simplePrimeppMultiGpu.cpp:      CHK_CUDA( cudaSetDevice(int(i)) );
    ./simplePrimeppMultiGpu/simplePrimeppMultiGpu.cpp:      CHK_CUDA( cudaSetDevice(int(i)) );
    ./simplePrimeppMultiGpu/simplePrimeppMultiGpu.cpp:      CHK_CUDA( cudaSetDevice(int(i)) );
    delta:OptiX_370b2_sdk blyth$ 
    delta:OptiX_370b2_sdk blyth$ 
    delta:OptiX_370b2_sdk blyth$ 
    delta:OptiX_370b2_sdk blyth$ pwd
    /usr/local/env/cuda/OptiX_370b2_sdk



/usr/local/env/cuda/OptiX_370b2_sdk/ocean/ocean.cpp::

    340     //
    341     // Setup cufft state
    342     //
    343 
    344     const unsigned int fft_input_size  = FFT_WIDTH * FFT_HEIGHT * sizeof(float2);
    345 
    346     m_context->launch( 0, 0 );
    ///
    ///     presumably ensures OptiX is first to setup CUDA context 
    ///
    347 
    348     m_cuda_device = OptiXDeviceToCUDADevice( m_context, 0 );
    ///
    ///     helper method: OptiX ordinal 0 -> CUDA ordinal 
    ///    
    349 
    350     if ( m_cuda_device < 0 ) {
    351       std::cerr << "OptiX device 0 must be a valid CUDA device number.\n";
    352       exit(1);
    353     }
    354 
    355     // output the CUFFT results directly into Optix buffer
    356     cudaSetDevice(m_cuda_device);
    357 
    358     cutilSafeCall( cudaMalloc( reinterpret_cast<void**>( &m_d_h0 ), fft_input_size ) );
    ///
    ///     plain CUDA allocation of space on device 
    ///
    359 
    360     m_h_h0      = new float2[FFT_WIDTH * FFT_HEIGHT];
    361     generateH0( m_h_h0 );
    /// 
    ///     host side generation, but it didnt have to be
    ///
    362 
    363     cutilSafeCall( cudaMemcpy( m_d_h0, m_h_h0, fft_input_size, cudaMemcpyHostToDevice) );
    ///
    ///     host to device copy  
    364 
    365     memcpy( m_h0_buffer->map(), m_h_h0, fft_input_size );
    366     m_h0_buffer->unmap();
    ///
    ///    copy from host into OptiX buffer
    ///
    367 
    368     // Finalize
    369     m_context->validate();
    370     m_context->compile();





OptiX and OpenGL interop : OptiX depth buffer calculation
------------------------------------------------------------

* http://rickarkin.blogspot.tw/2012/03/optix-is-ray-tracing-framework-it-can.html


Depth buffer combination may be the most important and a bit complicated. As a
ray tracing engine, Optix need not to do depth buffer test, so one can only
find the rtIntersectionDistance, which means the distance from the ray origin
to current ray-surface intersection point. So handily generate an OpenGL
compliant depth buffer is the first problem. A useful reference is
http://www.songho.ca/opengl/gl_projectionmatrix.html

My realization of the depth value construction is  attached as below::

    // eyeDist:  distance from eye to the intersection point.
    // n:           near clipping plane
    // f:            far clipping plane
    __device__ float computeClipDepth( float eyeDist, float n, float f )
    {
        float clipDepth = (f+n)/(f-n) - (1/eyeDist)*2.0f*f*n/(f-n);
        clipDepth = clipDepth*0.5 + 0.5f;
        return clipDepth;
    }

The second problem is to use the generated depth buffer of Optix into OpenGL.
Actually it is totally OpenGL operations. But maybe its not a daily used
process like draw a triangle or shading a scene object, so there is little
resource could be found on the web.  My realization of the depth value
construction is  also attached as below, where depthImg contains per pixel
depth value, coloredImg contains per pixel color value.::

    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // save buffer bit attribs to stack 
    glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);            // save client attrib to stack
    //
    // above lines prep for changing attribs by saving current ones
    //
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_FALSE);    
    glWindowPos2i(0, 0);  // specify the raster position in window coordinates for pixel operations
    glDrawPixels(w, h, GL_RGBA , GL_FLOAT, coloredImg);

    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_ALWAYS);
    //
    // specify the value used for depth buffer comparisons
    // 
    //    GL_LESS
    //         Passes if the incoming depth value is less than the stored depth value.
    //    GL_ALWAYS
    //         Always passes.  (unconditionally write to depth buffer)
    //  

    glWindowPos2i(0, 0);
    glDrawPixels(w, h, GL_DEPTH_COMPONENT , GL_FLOAT, depthImg);

    //
    //  restore initial attribute state 
    //
    glPopClientAttrib();
    glPopAttrib(); // GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT


OptiX talks
------------

* http://on-demand.gputechconf.com/gtc/2012/presentations/SS107-GPU-Ray-Tracing-OptiX.pdf

  Some details on multi GPU interop and avoiding unnecessary copying 


OptiX without OpenGL
----------------------

::

    delta:bin blyth$ ./sample7 -f out.ppm
    delta:bin blyth$ libpng-
    delta:bin blyth$ cat out.ppm | libpng-wpng > out.png 
    Encoding image data...
    Done.
    delta:bin blyth$ open out.png

 
OptiX and SLI handling of multiple GPUs
----------------------------------------

* http://en.wikipedia.org/wiki/Scalable_Link_Interface
* http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/

OptiX doesnt like SLI, switch it off.

::

   export CUDA_VISIBLE_DEVICES=0,1,2



OptiX OpenGL interop with PBO
-------------------------------

* http://rickarkin.blogspot.tw/2012/03/use-pbo-to-share-buffer-between-cuda.html

OptiX OpenGL interop 
---------------------

* :google:`OptiX OpenGL interop`

OptiX_Programming_Guide_3.7.0.pdf
-------------------------------------

OpenGL Interop
~~~~~~~~~~~~~~~~

OptiX supports interop for:

* OpenGL buffer objects
* textures
* render buffers

OpenGL buffer objects can be read and written by OptiX program
objects, whereas textures and render buffers can only be read.

Buffer Objects
~~~~~~~~~~~~~~

OpenGL buffer objects like PBOs and VBOs can be encapsulated for use in OptiX
with rtBufferCreateFromGLBO. The resulting buffer is a reference only to the
OpenGL data; the size of the OptiX buffer as well as the format have to be set
via rtBufferSetSize and rtBufferSetFormat. 
When the OptiX buffer is destroyed, the state of the OpenGL buffer object is unaltered. 
Once an OptiX buffer is created, the original GL buffer object is immutable, 
meaning the properties of the GL object like its size cannot be changed 
while registered with OptiX. However, it is still possible to read and write 
to the GL buffer object using the appropriate GL functions. 
If it is necessary to change properties of an object, first call rtBufferGLUnregister 
before making changes. After the changes are made the object has to be 
registered again with rtBufferGLRegister.
This is necessary to allow OptiX to access the objects data again. Registration
and unregistration calls are expensive and should be avoided if possible.






Caveats
----------

* https://devtalk.nvidia.com/default/topic/751906/?comment=4240594
  
  * rtPrintf not printf


OptiX Usage Examples
---------------------

* https://code.google.com/p/hybrid-rendering-thesis/source/browse/trunk/src/Raytracer/OptixRender.cpp?r=44

* https://github.com/keithroe/Legion/blob/master/src/Legion/Renderer/OptiXScene.cpp

  * CMakeLists.txt CUDA macro usage that stuffs ptx into libraries 

* https://github.com/pspkzar/OptiXRenderer/blob/master/src/OptixRenderer.cpp

* https://github.com/nvpro-samples/gl_optix_composite
* https://github.com/nvpro-samples/gl_optix_composite/blob/master/shaders/optix_triangle_mesh.cu

  * texture lookup example


* http://graphicsrunner.blogspot.tw/2011/03/instant-radiosity-using-optix-and.html


OptiX with GLFW
-----------------

* https://code.google.com/p/hybrid-rendering-thesis/source/browse/trunk/glfw_optix/src/main.cpp

See hrt-



nvfx : Generic Effect system for Graphic APIs, including OpenGL and OptiX
---------------------------------------------------------------------------

nvFx is a new approach for compositing shaders and compute kernels together,
using an API-agnostic description of effects for objects materials and scene
management (post-processing, management of rendering passes).

* **curious how the OptiX side of things is implemented**

  * specifically how material params are fed to the OptiX programs


* :google:`nvfx`

* https://github.com/tlorach/nvFX

* http://lorachnroll.blogspot.tw/2013/07/nvfx-effect-system-on-top-of-many.html



Large Codebases Using OptiX
-----------------------------

* macrosim-


Version Switching
------------------

Use symbolic link for version switching::

    delta:Developer blyth$ ll
    total 8
    drwxr-xr-x   7 root  admin   238 Aug  7  2013 OptiX_301
    drwxr-xr-x   3 root  wheel   102 Jan 15  2014 NVIDIA
    drwxr-xr-x   7 root  admin   238 Dec 18 07:08 OptiX_370b2
    drwxr-xr-x  33 root  wheel  1190 Jan 15 08:46 ..
    lrwxr-xr-x   1 root  wheel     9 Jan 22 11:27 OptiX -> OptiX_301
    drwxr-xr-x   6 root  wheel   204 Jan 22 11:27 .




Samples 
-------

::

   open file:///Developer/OptiX/SDK/NVIDIA-OptiX-SDK-samples.html


Above references a missing sample::

   file:///Developer/OptiX/SDK/collision/


* /Developer/OptiX/SDK-precompiled-samples/sample6.app

  * ray traced geometry of a cow

* /Developer/OptiX/SDK-precompiled-samples/shadeTree.app

  * Christmas decorations 



Building samples including sutil library 
-------------------------------------------


::

    delta:OptiX blyth$ optix-name
    OptiX_370b2
    delta:OptiX blyth$ optix-samples-get-all   # copy samples to writable dir
    delta:OptiX blyth$ optix-samples-cmake
    ...
    -- Found CUDA: /usr/local/cuda (Required is at least version "2.3") 
    -- Found OpenGL: /System/Library/Frameworks/OpenGL.framework  
    -- Found GLUT: -framework GLUT  
    Cannot find Cg, hybridShadows will not be built
    Cannot find Cg.h, hybridShadows will not be built
    Disabling hybridShadows, which requires glut and opengl and Cg.
    Cannot find Cg, isgReflections will not be built
    Cannot find Cg.h, isgReflections will not be built
    Disabling isgReflections, which requires glut and opengl and Cg.
    Cannot find Cg, isgShadows will not be built
    Cannot find Cg.h, isgShadows will not be built
    Disabling isgShadows, which requires glut and opengl and Cg.
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/env/cuda/OptiX_370b2_sdk_install

    delta:OptiX blyth$ optix-samples-make





Path to SAMPLES_PTX_DIR gets compiled in
-------------------------------------------

::

    delta:SDK blyth$ find . -name '*.*' -exec grep -H SAMPLES_PTX_DIR {} \;
    ./CMakeLists.txt:set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx" CACHE PATH "Path to where the samples look for the PTX code.")
    ./CMakeLists.txt:set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})
    ./CMakeLists.txt:  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
    ./sampleConfig.h.in:#define SAMPLES_PTX_DIR "@SAMPLES_PTX_DIR@"
    ./sutil/sutil.c:  dir = getenv( "OPTIX_SAMPLES_PTX_DIR" );
    ./sutil/sutil.c:  if( dirExists(SAMPLES_PTX_DIR) )
    ./sutil/sutil.c:    return SAMPLES_PTX_DIR;



OptiX-3.7.0-beta2
-------------------

* need to register with NVIDIA OptiX developer program to gain access 

Package installs into same place as 301::

    delta:Contents blyth$ pwd
    /Volumes/NVIDIA-OptiX-SDK-3.7.0-mac64/NVIDIA-OptiX-SDK-3.7.0-mac64.pkg/Contents
    delta:Contents blyth$ lsbom Archive.bom | head -5
    .   40755   501/0
    ./Developer 40755   501/0
    ./Developer/OptiX   40755   0/80
    ./Developer/OptiX/SDK   40755   0/80
    ./Developer/OptiX/SDK/CMake 40755   0/80

So move that aside::

    delta:Developer blyth$ sudo mv OptiX OptiX_301


* all precompiled samples failing 

::

    terminating with uncaught exception of type optix::Exception: Invalid context

    8   libsutil.dylib                  0x000000010f8b71d6 optix::Handle<optix::ContextObj>::create() + 150
    9   libsutil.dylib                  0x000000010f8b5b1b SampleScene::SampleScene() + 59
    10  libsutil.dylib                  0x000000010f8a6a52 MeshScene::MeshScene(bool, bool, bool) + 34
    11                                  0x000000010f870885 MeshViewer::MeshViewer() + 21


    delta:SDK-precompiled-samples blyth$ open ocean.app
    LSOpenURLsWithRole() failed with error -10810 for the file /Developer/OptiX/SDK-precompiled-samples/ocean.app.
    delta:SDK-precompiled-samples blyth$ 


    8   libsutil.dylib                  0x000000010e1141d6 optix::Handle<optix::ContextObj>::create() + 150
    9   libsutil.dylib                  0x000000010e112b1b SampleScene::SampleScene() + 59
    10                                  0x000000010e0d793c WhirligigScene::WhirligigScene(GLUTDisplay::contDraw_E) + 28



::

    delta:SDK-precompiled-samples blyth$ ./sample3
    OptiX 3.7.0
    Number of Devices = 1

    Device 0: GeForce GT 750M
      Compute Support: 3 0
      Total Memory: 2147024896 bytes
      Clock Rate: 925500 kilohertz
      Max. Threads per Block: 1024
      SM Count: 2
      Execution Timeout Enabled: 1
      Max. HW Texture Count: 128
      TCC driver enabled: 0
      CUDA Device Ordinal: 0

    Constructing a context...
    OptiX Error: Invalid context
    (/Volumes/DATA/teamcity/agent/work/ad29186266c461fa/sw/wsapps/raytracing/rtsdk/rel3.7/samples/sample3/sample3.c:96)
    delta:SDK-precompiled-samples blyth$ 

::

     95   printf("Constructing a context...\n");
     96   RT_CHECK_ERROR(rtContextCreate(&context));
     97 



This is with

* CUDA Driver Version: 5.5.47
* GPU Driver Version: 8.26.26 310.40.45f01



OptiX 301 Install issues 
--------------------------

* attempting to open pkg complains of unidentified developer

  * /Volumes/NVIDIA-OptiX-SDK-3.0.1-mac64/NVIDIA-OptiX-SDK-3.0.1-mac64.pkg

* make exception in `SysPrefs > Security & Privacy > General` 

  * hmm mavericks 10.9.4 "open anyway" doesnt work 
  * authenticate and change to from "anywhere" 


::

    delta:~ blyth$ optix-cmake
    ...
    Specified C compiler /usr/bin/cc is not recognized (gcc, icc).  Using CMake defaults.
    Specified CXX compiler /usr/bin/c++ is not recognized (g++, icpc).  Using CMake defaults.
    CMake Warning at CMake/ConfigCompilerFlags.cmake:195 (message):
      Unknown Compiler.  Disabling SSE 4.1 support
    Call Stack (most recent call first):
      CMakeLists.txt:116 (include)


    -- Found CUDA: /usr/local/cuda (Required is at least version "2.3") 
    -- Found OpenGL: /System/Library/Frameworks/OpenGL.framework  
    -- Found GLUT: -framework GLUT  
    Cannot find Cg, hybridShadows will not be built
    Cannot find Cg.h, hybridShadows will not be built
    Disabling hybridShadows, which requires glut and opengl and Cg.
    Cannot find Cg, isgShadows will not be built
    Cannot find Cg.h, isgShadows will not be built
    Disabling isgShadows, which requires glut and opengl and Cg.
    Cannot find Cg, isgReflections will not be built
    Cannot find Cg.h, isgReflections will not be built
    Disabling isgReflections, which requires glut and opengl and Cg.
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/env/cuda/optix301
    delta:optix301 blyth$ 


List the samples::

    delta:optix301 blyth$ optix-make help

All giving error::

    delta:optix301 blyth$ optix-make sample6
    [  7%] Building NVCC ptx file lib/ptx/cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx
    clang: error: unsupported option '-dumpspecs'
    clang: error: no input files
    CMake Error at cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx.cmake:200 (message):
      Error generating
      /usr/local/env/cuda/optix301/lib/ptx/cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx


    make[3]: *** [lib/ptx/cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx] Error 1
    make[2]: *** [sutil/CMakeFiles/sutil.dir/all] Error 2
    make[1]: *** [sample6/CMakeFiles/sample6.dir/rule] Error 2
    make: *** [sample6] Error 2
    delta:optix301 blyth$ 

Seems that nvcc is running clang internally with non existing option::

    delta:optix301 blyth$ /usr/local/cuda/bin/nvcc -M -D__CUDACC__ /Developer/OptiX/SDK/cuda/triangle_mesh_small.cu -o /usr/local/env/cuda/optix301/sutil/CMakeFiles/cuda_compile_ptx.dir/__/cuda/cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx.NVCC-depend -ccbin /usr/bin/cc -m64 -DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA --use_fast_math -U__BLOCKS__ -DNVCC -I/usr/local/cuda/include -I/Developer/OptiX/include -I/Developer/OptiX/SDK/sutil -I/Developer/OptiX/include/optixu -I/usr/local/env/cuda/optix301 -I/usr/local/cuda/include -I/System/Library/Frameworks/GLUT.framework/Headers -I/Developer/OptiX/SDK/sutil -I/Developer/OptiX/SDK/cuda
    clang: error: unsupported option '-dumpspecs'
    clang: error: no input files
    delta:optix301 blyth$ 


cmake debug
~~~~~~~~~~~~~

* added "--verbose"
* adding "-ccbin /usr/bin/clang" gets past the "--dumpspecs" failure, now get

    nvcc fatal   : redefinition of argument 'compiler-bindir'


* /Developer/OptiX/SDK/CMake/FindCUDA/run_nvcc.cmake::

    108 # Any -ccbin existing in CUDA_NVCC_FLAGS gets highest priority
    109 list( FIND CUDA_NVCC_FLAGS "-ccbin" ccbin_found0 )
    110 list( FIND CUDA_NVCC_FLAGS "--compiler-bindir" ccbin_found1 )
    111 if( ccbin_found0 LESS 0 AND ccbin_found1 LESS 0 )
    112   if (CUDA_HOST_COMPILER STREQUAL "$(VCInstallDir)bin" AND DEFINED CCBIN)
    113     set(CCBIN -ccbin "${CCBIN}")
    114   else()
    115     set(CCBIN -ccbin "${CUDA_HOST_COMPILER}")
    116   endif()
    117 endif()
     
    * http://public.kitware.com/Bug/view.php?id=13674


cmake fix
~~~~~~~~~~~~~~


Kludge the cmake::

    delta:FindCUDA blyth$ sudo cp run_nvcc.cmake run_nvcc.cmake.original
    delta:FindCUDA blyth$ pwd
    /Developer/OptiX/SDK/CMake/FindCUDA

Turns out not to be necessary, the cmake flag does the trick::

   cmake $(optix-dir) -DCUDA_NVCC_FLAGS="-ccbin /usr/bin/clang"
    

* :google:`cuda 5.5 clang`
* http://stackoverflow.com/questions/19351219/cuda-clang-and-os-x-mavericks
* http://stackoverflow.com/questions/12822205/nvidia-optix-geometrygroup


Check Optix Raytrace Speed on DYB geometry
--------------------------------------------

::

    In [3]: v=np.load(os.path.expandvars("$DAE_NAME_DYB_CHROMACACHE_MESH/vertices.npy"))

    In [4]: v
    Out[4]: 
    array([[ -16585.725, -802008.375,   -3600.   ],
           [ -16039.019, -801543.125,   -3600.   ],
           [ -15631.369, -800952.188,   -3600.   ],
           ..., 
           [ -14297.924, -801935.812,  -12110.   ],
           [ -14414.494, -801973.438,  -12026.   ],
           [ -14414.494, -801973.438,  -12110.   ]], dtype=float32)

    In [5]: v.shape
    Out[5]: (1216452, 3)

    In [6]: t = np.load(os.path.expandvars("$DAE_NAME_DYB_CHROMACACHE_MESH/triangles.npy"))
    In [7]: t.shape
    Out[7]: (2402432, 3)
    In [8]: t.max()
    Out[8]: 1216451
    In [9]: t.min()
    Out[9]: 0


Write geometry in obj format::

    In [11]: fp = file("/tmp/dyb.obj", "w")
    In [12]: np.savetxt(fp, v, fmt="v %.18e %.18e %.18e")
    In [13]: np.savetxt(fp, t, fmt="f %d %d %d")
    In [14]: fp.close()

Geometry appears mangled, as obj format does not handle Russian doll geometry, 
but the optix raytrace is interactive (unless some trickery being used, that is 
greatly faster than chroma raytrace). Fast enough to keep me interested::

    ./sample6 --cache --obj /tmp/dyb.obj --light-scale 5


How to load COLLADA into OptiX ?
-----------------------------------

* nvidia Scenix looks abandoned

* plumped for assimp following example of oppr- example, see assimp- assimptest- raytrace--

* oppr- converts ASSIMP imported mesh into OptiX geometry::

    delta:OppositeRenderer blyth$ find . -name '*.cpp' -exec grep -H getSceneRootGroup {} \;
    ./RenderEngine/renderer/OptixRenderer.cpp:        m_sceneRootGroup = scene.getSceneRootGroup(m_context);
    ./RenderEngine/scene/Cornell.cpp:optix::Group Cornell::getSceneRootGroup(optix::Context & context)
    ./RenderEngine/scene/Scene.cpp:optix::Group Scene::getSceneRootGroup( optix::Context & context )
    delta:OppositeRenderer blyth$ 


OptiX Tutorial
---------------

* http://docs.nvidia.com/gameworks/content/gameworkslibrary/optix/optix_quickstart.htm

Tutorials gt 5 asserting in rtContextCompile::

    delta:bin blyth$ ./tutorial -T 5 
    OptiX Error: Unknown error (Details: Function "RTresult _rtContextCompile(RTcontext_api*)" caught exception: Assertion failed: [1312612])

Binary search reveals the culprit to be the *sin(phi)*::

     74 rtTextureSampler<float4, 2> envmap;
     75 RT_PROGRAM void envmap_miss()
     76 {
     77   float theta = atan2f( ray.direction.x, ray.direction.z );
     78   float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
     79   float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
     80   float v     = 0.5f * ( 1.0f + sin(phi) );
     81   // the "sin" above causing an assert with OptiX_301 CUDA 5.5 without --use_fast_math 
     82   prd_radiance.result = make_float3( tex2D(envmap, u, v) );
     83 } 

* https://devtalk.nvidia.com/default/topic/559505/apparently-an-unexplicable-error/

Resolved by adding *--use_fast_math* to the cmake commandline setting CUDA_NVCC_FLAGS::

   cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) -DCUDA_NVCC_FLAGS="-ccbin /usr/bin/clang --use_fast_math " "$(optix-sdir)"

After a few crashes like the above observe GPU memory to be almost full
and attempts to run anything on the GPU fail with a system 
exception report. To free up some GPU memory sleep/revive the machine::

    delta:bin blyth$ cu
    timestamp                Fri Jan 23 10:44:11 2015
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              2.1G
    memory free              51.4M
    delta:bin blyth$ 


f64 check
-----------

::

    delta:raytrace blyth$ grep f64 *.ptx
    MeshViewer_generated_TriangleMesh.cu.ptx:   .target sm_10, map_f64_to_f32
    MeshViewer_generated_TriangleMesh.cu.ptx:   ld.global.f32   %f64, [ray+28];
    MeshViewer_generated_TriangleMesh.cu.ptx:   set.lt.u32.f32  %r30, %f64, %f58;
    MeshViewer_generated_material1.cu.ptx:  .target sm_10, map_f64_to_f32
    RayTrace_generated_TriangleMesh.cu.ptx: .target sm_10, map_f64_to_f32
    RayTrace_generated_TriangleMesh.cu.ptx: ld.global.f32   %f64, [ray+28];
    RayTrace_generated_TriangleMesh.cu.ptx: set.lt.u32.f32  %r30, %f64, %f58;
    RayTrace_generated_material0.cu.ptx:    .target sm_10, map_f64_to_f32
    RayTrace_generated_material1.cu.ptx:    .target sm_10, map_f64_to_f32
    RayTrace_generated_tutorial0.cu.ptx:    .target sm_10, map_f64_to_f32
    RayTrace_generated_tutorial0.cu.ptx:    cvt.sat.f32.f32     %f64, %f63;
    RayTrace_generated_tutorial0.cu.ptx:    mul.f32     %f66, %f64, %f65;
    delta:raytrace blyth$ 


Following updating CUDA from 5.5 to 6.5 get
---------------------------------------------

While still using OptiX301::

    delta:sample3 blyth$ raytrace-v -n
    [ 19%] Built target AssimpGeometryTest
    Scanning dependencies of target MeshViewer
    [ 22%] Building CXX object CMakeFiles/MeshViewer.dir/MeshViewer.cpp.o
    Linking CXX executable MeshViewer
    [ 61%] Built target MeshViewer
    [100%] Built target RayTrace
    dyld: Library not loaded: @rpath/libcudart.dylib
      Referenced from: /Developer/OptiX/lib64/liboptix.1.dylib
      Reason: Incompatible library version: liboptix.1.dylib requires version 1.1.0 or later, but libcudart.5.5.dylib provides version 0.0.0
    Trace/BPT trap: 5
    delta:sample3 blyth$ 


But with the beta OptiX_370b2 the invalid context issue is gone::

	delta:SDK-precompiled-samples blyth$ ./sample3
	OptiX 3.7.0
	Number of Devices = 1

	Device 0: GeForce GT 750M
	  Compute Support: 3 0
	  Total Memory: 2147024896 bytes
	  Clock Rate: 925500 kilohertz
	  Max. Threads per Block: 1024
	  SM Count: 2
	  Execution Timeout Enabled: 1
	  Max. HW Texture Count: 128
	  TCC driver enabled: 0
	  CUDA Device Ordinal: 0

	Constructing a context...
	  Created with 1 device(s)
	  Supports 2147483647 simultaneous textures
	  Free memory:
	    Device 0: 1099292672 bytes


OptiX cmake claims needs 2.8.8 by appears to build ok with 2.6.4
-------------------------------------------------------------------

::

    CMake Error at CMakeLists.txt:82 (cmake_minimum_required):
      CMake 2.8.8 or higher is required.  You are running version 2.6.4

::

    cmake-samples-cmake-kludge  # inplace edit the version requirement



Each OptiX release requires a different driver release
----------------------------------------------------------

sample7 works with Optix370 (new driver 340.65)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    -bash-4.1$ LD_LIBRARY_PATH=. ./sample7 -f out.ppm
    -bash-4.1$ ll *.ppm
    -rw-r--r-- 1 blyth dyw 2211856 Feb  9 11:53 out.ppm
    -bash-4.1$ date
    Mon Feb  9 11:54:10 CST 2015
    -bash-4.1$ optix-name
    NVIDIA-OptiX-SDK-3.7.0-linux64
    -bash-4.1$ 

    -bash-4.1$ cat /proc/driver/nvidia/version
    NVRM version: NVIDIA UNIX x86_64 Kernel Module  340.65  Tue Dec  2 09:50:34 PST 2014
    GCC version:  gcc version 4.4.7 20120313 (Red Hat 4.4.7-4) (GCC) 


sample7 fails with OptiX 363 (old driver)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

	-bash-4.1$ ./sample7 -f out.ppm
	OptiX Error: Invalid context (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Unable to set the CUDA device., [3735714])
	-bash-4.1$ 
	-bash-4.1$ pwd
	/dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn/env/cuda/NVIDIA-OptiX-SDK-3.6.3-linux64_sdk_install/bin
	-bash-4.1$ 


sample7 works with OptiX 351 (old driver)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    -bash-4.1$ LD_LIBRARY_PATH=. ./sample7 -f sample7.ppm

    delta:~ blyth$ scp L6:/dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn/OptiX/SDK-precompiled-samples/sample7.ppm .
    delta:~ blyth$ libpng-
    delta:~ blyth$ cat sample7.ppm | libpng-wpng > sample7.png
    Encoding image data...
    Done.
    delta:~ blyth$ open sample7.png



Benchmark
------------

Quick look at scaling with GPU cores, close to linear 

* CUDA_VISIBLE_DEVICES=0
* CUDA_VISIBLE_DEVICES=1
* unset 

* K20m 2496 cores each,  total 4992 cores

* 750m 384 cores 

::

    In [2]: 4992./384.
    Out[2]: 13.0

    In [3]: 345.154/28.03
    Out[3]: 12.313735283624688

    In [4]: 49.1/28.03
    Out[4]: 1.7516946129147342


1,1,2 K20m::


    Time to load geometry: 5.05531 s.
    Time to compile kernel: 0.220482 s.
    Time to build AS      : 0.162403 s.
    PREPROCESS: MeshViewer | compile 0.000169992 sec | accelbuild 0.0002141 sec
    BENCHMARK: MeshViewer | 20.3645 fps | 1 (warmup) | 2 (timed) | 0.0982101 sec | 49.105 ms/f 

    Time to load geometry: 4.99532 s.
    Time to compile kernel: 0.218149 s.
    Time to build AS      : 0.160232 s.
    PREPROCESS: MeshViewer | compile 0.000129938 sec | accelbuild 0.000213146 sec
    BENCHMARK: MeshViewer | 20.354 fps | 1 (warmup) | 2 (timed) | 0.0982609 sec | 49.1304 ms/f 

    Time to load geometry: 4.9231 s.
    Time to compile kernel: 0.22342 s.
    Time to build AS      : 0.312235 s.
    PREPROCESS: MeshViewer | compile 0.000135899 sec | accelbuild 0.000245094 sec
    BENCHMARK: MeshViewer | 35.669 fps | 1 (warmup) | 2 (timed) | 0.056071 sec | 28.0355 ms/f 


1 750m::

    Time to load geometry: 3.05554 s.
    Time to compile kernel: 0.160299 s.
    Time to build AS      : 0.286529 s.
    PREPROCESS: MeshViewer | compile 9.70364e-05 sec | accelbuild 0.000139952 sec
    BENCHMARK: MeshViewer | 2.89726 fps | 1 (warmup) | 2 (timed) | 0.690308 sec | 345.154 ms/f 









EOU
}



optix-export(){
   export OPTIX_SDK_DIR=$(optix-sdk-dir)
   export OPTIX_INSTALL_DIR=$(optix-install-dir)
   export OPTIX_SAMPLES_INSTALL_DIR=$(optix-samples-install-dir)
}

optix-fold(){ 
   case $NODE_TAG in 
      D)  echo /Developer ;;
      G1) echo $(local-base) ;;
      LT) echo /home/ihep/data/repo/opticks ;;
     GTL) echo /afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC6/GPU/20150723 ;;
      *) echo $(local-base) ;;
   esac
}
optix-prefix(){      echo $(optix-fold)/OptiX ; }
optix-dir(){         echo $(optix-fold)/OptiX/SDK ; }
optix-sdk-dir-old(){ echo $(optix-fold)/OptiX_301/SDK ; }
optix-sdk-dir(){     echo $(optix-fold)/OptiX/SDK ; }
optix-download-dir(){ echo $(local-base)/env/cuda ; }
optix-bdir(){         echo $(local-base)/env/cuda/$(optix-name) ; }
optix-install-dir(){ echo $(dirname $(optix-sdk-dir)) ; }
optix-idir(){        echo $(dirname $(optix-sdk-dir))/include ; }
optix-sdir(){        echo $(opticks-home)/optix ; }
optix-samples-src-dir(){     echo $(local-base)/env/cuda/$(optix-name)_sdk ; }
optix-samples-install-dir(){ echo $(local-base)/env/cuda/$(optix-name)_sdk_install ; }

optix-samples-scd(){   cd $(optix-samples-src-dir)/$1 ; }
optix-samples-cd(){    cd $(optix-samples-install-dir)/$1 ; }
optix-download-cd(){   cd $(optix-download-dir) ; }

optix-ftp(){ open https://ftpservices.nvidia.com ; }


optix-c(){   cd $(optix-dir); }
optix-cd(){  cd $(optix-dir); }
optix-bcd(){ cd $(optix-samples-install-dir); }
optix-scd(){ cd $(optix-sdir); }
optix-icd(){ cd $(optix-idir); }
optix-doc(){ cd $(optix-fold)/OptiX/doc ; }

optix-samples-find(){    optix-samples-cppfind $* ; }
optix-samples-cufind(){  find $(optix-samples-src-dir) -name '*.cu'  -exec grep ${2:--H} ${1:-rtReportIntersection} {} \; ;}
optix-samples-hfind(){   find $(optix-samples-src-dir) -name '*.h'   -exec grep ${2:--H} ${1:-rtReportIntersection} {} \; ;}
optix-samples-cppfind(){ find $(optix-samples-src-dir) -name '*.cpp' -exec grep ${2:--H} ${1:-rtReportIntersection} {} \; ;}
optix-find(){            find $(optix-idir)            -name '*.h'   -exec grep ${2:--H} ${1:-setMiss} {} \; ; }
optix-ifind(){           find $(optix-idir)            -name '*.h'   -exec grep ${2:--H} ${1:-setMiss} {} \; ; }

optix-x(){ find $(optix-dir) -name "*.${1}" -exec grep ${3:--H} ${2:-Sampler} {} \; ; }
optix-cu(){  optix-x cu  $* ; }
optix-cpp(){ optix-x cpp $* ; }
optix-h(){   optix-x h   $* ; }

optix-find(){
   optix-cu  $*
   optix-cpp $*
   optix-h  $*
}





optix-api-(){ echo $(optix-fold)/OptiX/doc/OptiX_API_Reference_$(optix-version).pdf ; }
optix-pdf-(){ echo $(optix-fold)/OptiX/doc/OptiX_Programming_Guide_$(optix-version).pdf ; }
optix-api(){ open $(optix-api-) ; }
optix-pdf(){ open $(optix-pdf-) ; }



optix-readlink(){ readlink $(optix-fold)/OptiX ; }
optix-name(){  echo ${OPTIX_NAME:-$(optix-readlink)} ; }
optix-jump(){    
   local iwd=$PWD
   local ver=${1:-301}
   cd $(optix-fold)
   sudo ln -sfnv OptiX_$ver OptiX 
   cd $iwd
}
optix-old(){   optix-jump 301 ; }
optix-beta(){  optix-jump 370b2 ; }

optix-linux-name(){
   case $1 in 
      351) echo NVIDIA-OptiX-SDK-3.5.1-PRO-linux64 ;;
      363) echo NVIDIA-OptiX-SDK-3.6.3-linux64 ;;
      370) echo NVIDIA-OptiX-SDK-3.7.0-linux64 ;;
   esac
}

optix-version(){
   case $(optix-name) in 
     OptiX_400)   echo 4.0.0 ;;
     OptiX_380)   echo 3.8.0 ;;
     OptiX_301)   echo 3.0.2 ;;
     OptiX_370b2) echo 3.7.0 ;;
  esac
}

optix-vernum(){
   case $(optix-name) in 
     OptiX_400)   echo 400 ;;
     OptiX_380)   echo 380 ;;
     OptiX_301)   echo 302 ;;
     OptiX_370b2) echo 370 ;;
  esac
}




optix-linux-jump(){
    local vers=${1:-351}
    local name=$(optix-linux-name $vers)
    [ -z "$name" ] && echo $msg version $vers unknown && type optix-linux-name && return 

    cd $(optix-fold)    
    ln -sfnv $name OptiX
}

   



optix-samples-names(){ cat << EON
CMakeLists.txt
sampleConfig.h.in
cuda
CMake
sample1
sample2
sample3
sample4
sample5
sample5pp
sample6
sample7
sample8
simpleAnimation
sutil
tutorial
materials
transparency
EON

## remember that after adding a name here, need to uncomment in the CMakeLists.txt for it to get built
}



optix-samples-get-all(){

   local src=$(optix-sdk-dir)
   local dst=$(optix-samples-src-dir)
 
   mkdir -p $dst

   echo $FUNCNAME copy all samples to somewhere writable 
   cp -R $src/* $dst/
 
}


optix-samples-get-selected(){
   local sdir=$(optix-samples-src-dir)
   mkdir -p $sdir

   local src=$(optix-sdk-dir)
   local dst=$sdir
   local cmd
   local name
   optix-samples-names | while read name ; do 

      if [ -d "$src/$name" ]
      then 
          if [ ! -d "$dst/$name" ] ; then 
              cmd="cp -r $src/$name $dst/"
          else
              cmd="echo destination directory exists already $dst/$name"
          fi
      elif [ -f "$src/$name" ] 
      then 
          if [ ! -f "$dst/$name" ] ; then 
              cmd="cp $src/$name $dst/$name"
          else
              cmd="echo destination file exists already $dst/$name"
          fi
      else
          cmd="echo src $src/$name missing"
      fi 
      #echo $cmd
      eval $cmd
   done
}


optix-cuda-nvcc-flags(){
    case $NODE_TAG in 
       D) echo -ccbin /usr/bin/clang --use_fast_math ;;
       *) echo --use_fast_math ;; 
    esac
}



#optix-samples-cmake-kludge(){
#    optix-samples-scd
#    grep cmake_minimum_required CMakeLists.txt 
#    perl -pi -e 's,2.8.8,2.6.4,' CMakeLists.txt 
#    grep cmake_minimum_required CMakeLists.txt 
#}


optix-samples-cmake(){
    local iwd=$PWD
    local bdir=$(optix-samples-install-dir)
    #rm -rf $bdir   # starting clean 
    mkdir -p $bdir
    optix-bcd
    cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) \
          -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
           "$(optix-samples-src-dir)"
    cd $iwd
}

optix-samples-make(){
    local iwd=$PWD
    optix-bcd
    make $* 
    cd $iwd
}




optix-samples-run(){
    local name=${1:-materials}
    optix-samples-make $name
    local cmd="$(optix-bdir)/bin/$name"
    echo $cmd
    eval $cmd
}

optix-tutorial(){
    local tute=${1:-10}

    optix-samples-make tutorial

    local cmd="$(optix-bdir)/bin/tutorial -T $tute --texture-path $(optix-sdk-dir)/tutorial/data"
    echo $cmd
    eval $cmd
}


optix-tutorial-cd(){
   cd $(optix-sdk-dir)/tutorial
}
optix-tutorial-vi(){
   vi $(optix-sdk-dir)/tutorial/*
}




optix-verbose(){
  export VERBOSE=1 
}
optix-unverbose(){
  unset VERBOSE
}



optix-check(){
/usr/local/cuda/bin/nvcc -ccbin /usr/bin/clang --verbose -M -D__CUDACC__ /Developer/OptiX/SDK/cuda/triangle_mesh_small.cu -o /usr/local/env/cuda/optix301/sutil/CMakeFiles/cuda_compile_ptx.dir/__/cuda/cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx.NVCC-depend -ccbin /usr/bin/cc -m64 -DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA --use_fast_math -U__BLOCKS__ -DNVCC -I/usr/local/cuda/include -I/Developer/OptiX/include -I/Developer/OptiX/SDK/sutil -I/Developer/OptiX/include/optixu -I/usr/local/env/cuda/optix301 -I/usr/local/cuda/include -I/System/Library/Frameworks/GLUT.framework/Headers -I/Developer/OptiX/SDK/sutil -I/Developer/OptiX/SDK/cuda
}



optix-check-2(){

cd /usr/local/env/cuda/OptiX_301/tutorial && /usr/bin/c++   -DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA -fPIC -O3 -DNDEBUG \
     -I/Developer/OptiX/include \
     -I/Users/blyth/env/cuda/optix/OptiX_301/sutil \
     -I/Developer/OptiX/include/optixu \
     -I/usr/local/env/cuda/OptiX_301 \
     -I/usr/local/cuda/include \
     -I/System/Library/Frameworks/GLUT.framework/Headers \
       -o /dev/null \
       -c /Users/blyth/env/cuda/optix/OptiX_301/tutorial/tutorial.cpp

}



optix-diff(){
   local name=${1:-sutil/MeshScene.h}
   local cmd="diff $(optix-sdk-dir-old)/$name $(optix-sdk-dir)/$name"
   echo $cmd
   eval $cmd
}

optix-rdiff(){
   local rel="sutil"
   local cmd="diff -r --brief $(optix-sdk-dir-old)/$rel $(optix-sdk-dir)/$rel"
   echo $cmd
   eval $cmd
}



optix-pkgname(){ echo NVIDIA-OptiX-SDK-$(optix-version)-mac64 ; }

optix-dmgpath()
{
    echo $(local-base)/env/cuda/$(optix-pkgname).dmg
}
optix-dmgpath-open()
{
    open $(optix-dmgpath)
}
optix-pkgpath()
{
    echo /Volumes/$(optix-pkgname)/$(optix-pkgname).pkg
}
optix-pkgpath-lsbom()
{
    #lsbom "$(pkgutil --bom "$(optix-pkgpath)")" 
    lsbom $(optix-pkgpath)/Contents/Archive.bom 
}




