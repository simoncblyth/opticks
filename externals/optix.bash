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



EON
}

optix-sfind(){    optix-samples-scd ; find . \( -name '*.cu' -or -name '*.h' -or -name '*.cpp'  \) -exec grep ${2:--H} "${1:-rtReport}" {} \; ; }

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



