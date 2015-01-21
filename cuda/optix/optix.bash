# === func-gen- : cuda/optix/optix fgp cuda/optix/optix.bash fgn optix fgh cuda/optix
optix-src(){      echo cuda/optix/optix.bash ; }
optix-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optix-src)} ; }
optix-vi(){       vi $(optix-source) ; }
optix-env(){      elocal- ; }
optix-usage(){ cat << EOU

::

    delta:~ blyth$ optix-cmake
    -- The C compiler identification is Clang 6.0.0
    -- The CXX compiler identification is Clang 6.0.0
    -- Check for working C compiler: /usr/bin/cc
    -- Check for working C compiler: /usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
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



Debug 

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




EOU
}
optix-bdir(){ echo $(local-base)/env/cuda/optix301 ; }
optix-sdir(){ echo $(env-home)/cuda/optix/optix301 ; }
optix-dir(){ echo /Developer/OptiX/SDK ; }
optix-sdk-dir(){ echo /Developer/OptiX/SDK ; }
optix-install-dir(){ echo $(dirname $(optix-sdk-dir)) ; }

optix-cd(){  cd $(optix-dir); }
optix-bcd(){ cd $(optix-bdir); }
optix-scd(){ cd $(optix-sdir); }

optix-mate(){ mate $(optix-dir) ; }


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
EON
}

optix-samples-get(){
   local sdir=$(optix-sdir)
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


optix-samples-cmake(){
   local bdir=$(optix-bdir)
   rm -rf $bdir   # starting clean everytime
  
   mkdir -p $bdir

   optix-bcd

   cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) -DCUDA_NVCC_FLAGS="-ccbin /usr/bin/clang" "$(optix-sdir)"
}





optix-export(){
   export OPTIX_SDK_DIR=$(optix-sdk-dir)
}


optix-cmake(){

   local bdir=$(optix-bdir)
   mkdir -p $bdir
   optix-bcd

   cmake $(optix-dir) -DCUDA_NVCC_FLAGS="-ccbin /usr/bin/clang"

}

optix-make(){
  optix-bcd
  export VERBOSE=1 
  make $* 
}


optix-check-notes(){ cat << EON
 

EON
}

optix-check(){
/usr/local/cuda/bin/nvcc -ccbin /usr/bin/clang --verbose -M -D__CUDACC__ /Developer/OptiX/SDK/cuda/triangle_mesh_small.cu -o /usr/local/env/cuda/optix301/sutil/CMakeFiles/cuda_compile_ptx.dir/__/cuda/cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx.NVCC-depend -ccbin /usr/bin/cc -m64 -DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA --use_fast_math -U__BLOCKS__ -DNVCC -I/usr/local/cuda/include -I/Developer/OptiX/include -I/Developer/OptiX/SDK/sutil -I/Developer/OptiX/include/optixu -I/usr/local/env/cuda/optix301 -I/usr/local/cuda/include -I/System/Library/Frameworks/GLUT.framework/Headers -I/Developer/OptiX/SDK/sutil -I/Developer/OptiX/SDK/cuda
}



optix-sample-name(){ echo ${OPTIX_SAMPLE_NAME:-sample6} ; }

optix-sample(){
   local name=$($FUNCNAME-name)
   local cmd="$(optix-bdir)/bin/$name $*"
   echo $cmd
   eval $cmd
}

