# === func-gen- : cuda/optix/optix fgp cuda/optix/optix.bash fgn optix fgh cuda/optix
optix-src(){      echo cuda/optix/optix.bash ; }
optix-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optix-src)} ; }
optix-vi(){       vi $(optix-source) ; }
optix-env(){      elocal- ; }
optix-usage(){ cat << EOU

NVIDIA OptiX Ray Trace Toolkit
================================== 

Resources
----------

* https://devtalk.nvidia.com/default/board/90/optix/


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




OptiX 301 Install issues 
--------------------------

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

* plumped for assimp following example of oppr- example, see assimp- assimptest-

* oppr- converts ASSIMP imported mesh into OptiX geometry::

    delta:OppositeRenderer blyth$ find . -name '*.cpp' -exec grep -H getSceneRootGroup {} \;
    ./RenderEngine/renderer/OptixRenderer.cpp:        m_sceneRootGroup = scene.getSceneRootGroup(m_context);
    ./RenderEngine/scene/Cornell.cpp:optix::Group Cornell::getSceneRootGroup(optix::Context & context)
    ./RenderEngine/scene/Scene.cpp:optix::Group Scene::getSceneRootGroup( optix::Context & context )
    delta:OppositeRenderer blyth$ 


/usr/local/env/cuda/optix/OppositeRenderer/OppositeRenderer/RenderEngine/scene/Scene.cpp::

    298 optix::Group Scene::getSceneRootGroup( optix::Context & context )
    299 {
    300     if(!m_intersectionProgram)
    301     {
    302         std::string ptxFilename = "TriangleMesh.cu.ptx";
    303         m_intersectionProgram = context->createProgramFromPTXFile( ptxFilename, "mesh_intersect" );
    304         m_boundingBoxProgram = context->createProgramFromPTXFile( ptxFilename, "mesh_bounds" );
    305     }
    306 
    307     //printf("Sizeof materials array: %d", materials.size());
    308 
    309     //QVector<optix::GeometryInstance> instances;
    310 
    311     // Convert meshes into Geometry objects
    312 
    313     QVector<optix::Geometry> geometries;
    314     for(unsigned int i = 0; i < m_scene->mNumMeshes; i++)
    315     {
    316         optix::Geometry geometry = createGeometryFromMesh(m_scene->mMeshes[i], context);
    317         geometries.push_back(geometry);
    318         //optix::GeometryInstance instance = getGeometryInstanceFromMesh(m_scene->mMeshes[i], context, materials);
    319         //instances.push_back(instance);
    320     }
    321 
    322     // Convert nodes into a full scene Group
    323 
    324     optix::Group rootNodeGroup = getGroupFromNode(context, m_scene->mRootNode, geometries, m_materials);
    ...
    342     optix::Acceleration acceleration = context->createAcceleration("Sbvh", "Bvh");
    343     rootNodeGroup->setAcceleration( acceleration );
    344     acceleration->markDirty();
    345     return rootNodeGroup;
    346 }
    ...
    348 optix::Geometry Scene::createGeometryFromMesh(aiMesh* mesh, optix::Context & context)
    349 {
    350     unsigned int numFaces = mesh->mNumFaces;
    351     unsigned int numVertices = mesh->mNumVertices;
    352 
    353     optix::Geometry geometry = context->createGeometry();
    354     geometry->setPrimitiveCount(numFaces);
    355     geometry->setIntersectionProgram(m_intersectionProgram);
    356     geometry->setBoundingBoxProgram(m_boundingBoxProgram);
    357 
    358     // Create vertex, normal and texture buffer
    359 
    360     optix::Buffer vertexBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
    361     optix::float3* vertexBuffer_Host = static_cast<optix::float3*>( vertexBuffer->map() );
    362 
    363     optix::Buffer normalBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
    364     optix::float3* normalBuffer_Host = static_cast<optix::float3*>( normalBuffer->map() );
    365 
    366     geometry["vertexBuffer"]->setBuffer(vertexBuffer);
    367     geometry["normalBuffer"]->setBuffer(normalBuffer);
    368 
    369     // Copy vertex and normal buffers
    370 
    371     memcpy( static_cast<void*>( vertexBuffer_Host ),
    372         static_cast<void*>( mesh->mVertices ),
    373         sizeof( optix::float3 )*numVertices);
    374     vertexBuffer->unmap();



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



EOU
}




optix-export(){
   export OPTIX_SDK_DIR=$(optix-sdk-dir)
   export OPTIX_INSTALL_DIR=$(optix-install-dir)
   export OPTIX_SAMPLES_INSTALL_DIR=$(optix-samples-install-dir)
}

optix-fold(){    echo /Developer ; }
optix-dir(){     echo $(optix-fold)/OptiX/SDK ; }
optix-sdk-dir(){ echo $(optix-fold)/OptiX/SDK ; }
optix-install-dir(){ echo $(dirname $(optix-sdk-dir)) ; }
optix-bdir(){ echo $(local-base)/env/cuda/$(optix-name) ; }
optix-sdir(){ echo $(env-home)/cuda/optix/$(optix-name) ; }
optix-samples-install-dir(){ echo $(optix-bdir) ; }

optix-cd(){  cd $(optix-dir); }
optix-bcd(){ cd $(optix-bdir); }
optix-scd(){ cd $(optix-sdir); }



optix-name(){   readlink $(optix-fold)/OptiX ; }
optix-jump(){    
   local iwd=$PWD
   local ver=${1:-301}
   cd $(optix-fold)
   sudo ln -sfnv OptiX_$ver OptiX 
   cd $iwd
}
optix-old(){   optix-jump 301 ; }
optix-beta(){  optix-jump 370b2 ; }


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
    local iwd=$PWD
    local bdir=$(optix-bdir)
    #rm -rf $bdir   # starting clean 
    mkdir -p $bdir
    optix-bcd
    cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) -DCUDA_NVCC_FLAGS="-ccbin /usr/bin/clang --use_fast_math " "$(optix-sdir)"
    cd $iwd
}

optix-samples-make(){
    local iwd=$PWD
    optix-bcd
    make $* 
    cd $iwd
}

optix-tutorial(){
    local tute=${1:-10}

    optix-samples-make tutorial

    local cmd="$(optix-bdir)/bin/tutorial -T $tute --texture-path $(optix-sdk-dir)/tutorial/data"
    echo $cmd
    eval $cmd
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



