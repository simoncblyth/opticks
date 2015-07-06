# === func-gen- : graphics/ggeoview/ggeoview fgp graphics/ggeoview/ggeoview.bash fgn ggeoview fgh graphics/ggeoview

ggv-(){   ggeoview- ; }
ggv-cd(){ ggeoview-cd ; }
ggv-i(){  ggeoview-install ; }
ggv--(){  ggeoview-depinstall ; }
ggv-lldb(){ ggeoview-lldb $* ; }

ggeoview-src(){      echo graphics/ggeoview/ggeoview.bash ; }
ggeoview-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ggeoview-src)} ; }
ggeoview-vi(){       vi $(ggeoview-source) ; }
ggeoview-usage(){ cat << EOU

GGeoView
==========

Start from glfwtest- and add in OptiX functionality from optixrap-

* NB raytrace- is another user of optixwrap- 


Usage tips
-----------

Thoughts on touch mode : OptiX single-ray-cast OR OpenGL depth buffer/unproject 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OptiX based touch mode, is not so useful operationally (although handy as a debug tool) as:

#. it requires to do an OptiX render before it can operate
#. will usually be using OpenGL rendering to see the geometry often with 
   clipping planes etc.. that only OpenGL knows about.  

Thus need an OpenGL depth buffer unproject approach too.


Low GPU memory running
~~~~~~~~~~~~~~~~~~~~~~~~~~

When GPU memory is low OptiX startup causes a crash, 
to run anyhow disable OptiX with::

    ggeoview-run --optixmode -1

To free up GPU memory restart the machine, or try sleep/unsleep and
exit applications including Safari, Mail that all use GPU memory. 
Observe that sleeping for ~1min rather than my typical few seconds 
frees almost all GPU memory.

Check available GPU memory with **cu** if less than ~512MB OptiX will
crash at startup::

    delta:optixrap blyth$ t cu
    cu is aliased to cuda_info.sh


Clipping Planes and recording frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    udp.py --cutnormal 1,0,0 --eye -2,0,0 --cutpoint 0,0,0
    udp.py --cutnormal 1,1,0 --cutpoint -0.1,-0.1,0


Although model frame coordinates are useful for 
intuitive data entry the fact that the meaning is relative
to the selected geometry makes them less 
useful as a way of recording a plane, 
so record planes in world frame coordinates.

This would allow to find the precise plane that 
halves a piece of geometry by selecting that
and providing a way to accept world planes, could 
use --cutplane x,y,x,w to skip the model_to_world 
conversion.

The same thinking applies to recording viewpoint bookmarks.


New way of interpolated photon position animation ?
----------------------------------------------------

See oglrap- for untested idea using geometry shaders alone.


Old way of doing interpolated photon position animation
-----------------------------------------------------------

* splayed out by maxsteps VBO

* recorded each photon step into its slot 

* pre-render CUDA time-presenter to find before and after 
  positions and interpolate between them writing into specific top slot 
  of the splay.


Problems:

* limited numbers of particles can be animated (perhaps 1000 or so)
  as approach multiplies storage by the max number of steps are kept

* most of the storage is empty, for photons without that many steps 

Advantages:

* splaying out allows CUDA to operate fully concurrently 
  with no atomics complexities, as every photon step has its place 
  in the structure 

* OpenGL can address and draw the VBO using fixed offsets/strides
  pointing at the interpolated slot, geometry shaders can be used to 
  amplify a point and momentum direction into a line


Package Dependencies Tree of GGeoView
--------------------------------------

* higher level repeated dependencies elided for clarity 

::

    NPY*   (~11 classes)
       Boost
       GLM         

    Cfg*  (~1 class)
       Boost 

    numpyserver*  (~7 classes)
       Boost.Asio
       ZMQ
       AsioZMQ
       Cfg*
       NPY*

    cudawrap* (~5 classes)
       CUDA



 
    GGeo*  (~22 classes)
       NPY*

    AssimpWrap* (~7 classes)
       Assimp
       GGeo* 

    OGLRap*  (~29 classes)
       GLEW
       GLFW
       ImGui
       AssimpWrap*
       Cfg*
       NPY*

    OptiXRap* (~7 classes)
       OptiX
       OGLRap*
       AssimpWrap*
       GGeo*    
   


Data Flow thru the app
-------------------------

* Gensteps NPY loaded from file (or network)

* main.NumpyEvt::setGenstepData 

  * determines num_photons
  * allocates NPY arrays for photons, records, sequence, recsel, phosel
    and characterizes content with MultiViewNPY 

* main.Scene::uploadEvt

  * gets genstep, photon and record renderers to upload their respective buffers 
    and translate MultiViewNPY into OpenGL vertex attributes

* main.Scene::uploadSelection

  * recsel upload
  * hmm currently doing this while recsel still all zeroes 

* main.OptiXEngine::initGenerate(NumpyEvt* evt)

  * populates OptiX context, using OpenGL buffer ids lodged in the NPY  
    to create OptiX buffers for each eg::

        m_genstep_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, genstep_buffer_id);

* main.OptiXEngine::generate, cu/generate.cu

  * fills OptiX buffers: photon_buffer, record_buffer, sequence_buffer

* main.Rdr::download(NPY*)

  * pullback to host NPY the VBO/OptiX buffers using Rdr::mapbuffer 
    Rdr::unmapbuffer to get void* pointers from OpenGL

    * photon, record and sequence buffers are downloaded

* main.ThrustArray::ThrustArray created for: sequence, recsel and phosel 

  * OptiXUtil::getDevicePtr devptr used to allow Thrust to access these OpenGL buffers 
    
* main.ThrustIdx indexes the sequence outputing into phosel and recsel

  * recsel is created from phosel using ThrustArray::repeat_to

* main.Scene::render Rdr::render for genstep, photon, record 

  * glBindVertexArray(m_vao) and glDrawArrays 
  * each renderer has a single m_vao which contains buffer_id and vertex attributes


Issue: recsel changes not seen by OpenGL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The zeroed recsel buffer was uploaded early, it was modified
with Thrust using the below long pipeline but the 
changes to the device buffer where not seen by OpenGL

* NumpyEvt create NPY
* Scene::uploadEvt, Scene::uploadSelection - Rdr::upload (setting buffer_id in the NPY)
* OptiXEngine::init (convert to OptiX buffers)
* OptiXUtil provides raw devptr for use by ThrustArray
* Rdr::render draw shaders do not see the changes to the recsel buffer 

Workaround by simplifying pipeline for non-conforming buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recsel, phosel do not conform to the pattern of other buffers 

* not needed by OptiX
* only needed in host NPY for debugging 
* phosel is populated on device by ThrustIdx::makeHistogram from the OptiX filled sequence buffer
* recsel is populated on device by ThrustArray::repeat_to on phosel 

Formerly had no way to get buffers into Thrust other than 
going through the full pipeline. Added capability to ThrustArray 
to allocate/resize buffers allowing simpler flow:

* NumpyEvt create NPY (recsel, phosel still created early on host, but they just serve as dimension placeholders)
* allocate recsel and phosel on device with ThrustArray(NULL, NPY dimensions), populate with ThrustIdx
* ThrustArray::download into the recsel and phosel NPY 
* Scene::uploadSelection to upload with OpenGL for use from shaders 

TODO: skip redundant Thrust download, OpenGL upload using CUDA/OpenGL interop ?



C++ library versions
----------------------

::

    delta:~ blyth$ otool -L $(ggeoview-;ggeoview-deps) | grep c++
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 60.0.0)



Pre-cook RNG Cache
-------------------

* currently the work number must precicely match the hardcoded 
  value used for OptiXEngine::setRngMax  

  * TODO: tie these together via envvar


::

    delta:ggeoview blyth$ ggeoview-rng-prep
    cuRANDWrapper::instanciate with cache enabled : cachedir /usr/local/env/graphics/ggeoview.build/lib/rng
    cuRANDWrapper::Allocate
    cuRANDWrapper::InitFromCacheIfPossible
    cuRANDWrapper::InitFromCacheIfPossible : no cache initing and saving 
    cuRANDWrapper::Init
     init_rng_wrapper sequence_index   0  thread_offset       0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   138.0750 ms 
    ...



Improvement
-------------

Adopt separate minimal VBO for animation 

* single vec4 (position, time) ? 
* no need for direction as see that from the interpolation, 
* polz, wavelength, ... keep these in separate full-VBO for detailed debug 
  of small numbers of stepped photons 


Does modern OpenGL have any features that allow a better way
--------------------------------------------------------------

* http://gamedev.stackexchange.com/questions/20983/how-is-animation-handled-in-non-immediate-opengl

  * vertex blend-based animation
  * vertex blending
  * use glVertexAttribPointer to pick keyframes, 
  * shader gets two "position" attributes, 
    one for the keyframe in front of the current 
    time and one for the keyframe after and a uniform that specifies 
    how much of a blend to do between them. 

hmm not so easy for photon simulation as they all are on their own timeline :
so not like traditional animation keyframes
 


glDrawArraysIndirect introduced in 4.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.opengl.org/wiki/History_of_OpenGL#OpenGL_4.1_.282010.29
* https://www.opengl.org/wiki/Vertex_Rendering#Indirect_rendering
* http://stackoverflow.com/questions/5047286/opengl-4-0-gpu-draw-feature
* https://www.opengl.org/registry/specs/ARB/draw_indirect.txt

Indirect rendering is the process of issuing a drawing command to OpenGL,
except that most of the parameters to that command come from GPU storage
provided by a Buffer Object.
The idea is to avoid the GPU->CPU->GPU round-trip; the GPU decides what range
of vertices to render with. All the CPU does is decide when to issue the
drawing command, as well as which Primitive is used with that command.

The indirect rendering functions take their data from the buffer currently
bound to the GL_DRAW_INDIRECT_BUFFER binding. Thus, any of these
functions will fail if no buffer is bound to that binding.

So can tee up a buffer of commands GPU side, following layout::

    void glDrawArraysIndirect(GLenum mode, const void *indirect);

    typedef  struct {
       GLuint  count;
       GLuint  instanceCount;
       GLuint  first;
       GLuint  baseInstance;   // MUST BE 0 IN 4.1
    } DrawArraysIndirectCommand;

Where each cmd is equivalent to::

    glDrawArraysInstancedBaseInstance(mode, cmd->first, cmd->count, cmd->instanceCount, cmd->baseInstance);

Similarly for indirect indexed drawing::

    glDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect);

    typedef  struct {
        GLuint  count;
        GLuint  instanceCount;
        GLuint  firstIndex;
        GLuint  baseVertex;
        GLuint  baseInstance;
    } DrawElementsIndirectCommand;

With each cmd equivalent to:: 

    glDrawElementsInstancedBaseVertexBaseInstance(mode, cmd->count, type,
      cmd->firstIndex * size-of-type, cmd->instanceCount, cmd->baseVertex, cmd->baseInstance);

* https://www.opengl.org/sdk/docs/man/html/glDrawElementsInstancedBaseVertex.xhtml


EOU
}


ggeoview-sdir(){ echo $(env-home)/graphics/ggeoview ; }
ggeoview-idir(){ echo $(local-base)/env/graphics/ggeoview ; }
ggeoview-bdir(){ echo $(ggeoview-idir).build ; }

#ggeoview-rng-dir(){ echo $(ggeoview-bdir)/lib/rng ; }  gets deleted too often for keeping RNG 
ggeoview-rng-dir(){ echo $(ggeoview-idir)/cache/rng ; }

ggeoview-ptx-dir(){ echo $(ggeoview-bdir)/lib/ptx ; }
ggeoview-rng-ls(){  ls -l $(ggeoview-rng-dir) ; }
ggeoview-ptx-ls(){  ls -l $(ggeoview-ptx-dir) ; }

ggeoview-scd(){  cd $(ggeoview-sdir); }
ggeoview-cd(){  cd $(ggeoview-sdir); }

ggeoview-icd(){  cd $(ggeoview-idir); }
ggeoview-bcd(){  cd $(ggeoview-bdir); }
ggeoview-name(){ echo GGeoView ; }

ggeoview-wipe(){
   local bdir=$(ggeoview-bdir)
   rm -rf $bdir
}
ggeoview-env(){     
    elocal- 
    optix-
    optix-export
}

ggeoview-cmake(){
   local iwd=$PWD

   local bdir=$(ggeoview-bdir)
   mkdir -p $bdir
  
   ggeoview-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(ggeoview-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(ggeoview-sdir)

   cd $iwd
}

ggeoview-make(){
   local iwd=$PWD

   ggeoview-bcd 
   make $*

   cd $iwd
}

ggeoview-install(){
   printf "********************** $FUNCNAME "
   ggeoview-make install
}

ggeoview-bin(){ echo $(ggeoview-idir)/bin/$(ggeoview-name) ; }


ggeoview-accelcache()
{
    ggeoview-export
    ls -l ${DAE_NAME_DYB/.dae}.*.accelcache
}
ggeoview-accelcache-rm()
{
    ggeoview-export
    rm ${DAE_NAME_DYB/.dae}.*.accelcache
}

ggeoview-rng-max()
{
   # maximal number of photons that can be handled
    echo $(( 1000*1000*3 ))
}

ggeoview-rng-prep()
{
   cudawrap-
   CUDAWRAP_RNG_DIR=$(ggeoview-rng-dir) CUDAWRAP_RNG_MAX=$(ggeoview-rng-max) $(cudawrap-ibin)
}



ggeoview-idpath()
{
   ggeoview-
   ggeoview-run --idpath 2>/dev/null 
}

ggeoview-steal-bookmarks()
{
   local idpath=$(ggeoview-idpath)
   cp ~/.g4daeview/dyb/bookmarks20141128-2053.cfg $idpath/bookmarks.ini
}


ggeoview-export()
{
   export-
   export-export

   export GGEOVIEW_GEOKEY="DAE_NAME_DYB"

   local q
   q="range:3153:12221"
   #q="range:3153:4814"     #  transition to 2 AD happens at 4814 
   #q="range:3153:4813"     #  this range constitutes full single AD
   #q="range:3161:4813"      #  push up the start to get rid of plain outer volumes, cutaway view: udp.py --eye 1.5,0,1.5 --look 0,0,0 --near 5000
   #q="index:5000"
   #q="index:3153,depth:25"
   #q="range:5000:8000"

   export GGEOVIEW_QUERY=$q
   export GGEOVIEW_CTRL=""
   export SHADER_DIR=$(ggeoview-sdir)/gl
   export SHADER_INCL_PATH=$(ggeoview-sdir)/gl:$(ggeoview-sdir)

   export RAYTRACE_PTX_DIR=$(ggeoview-ptx-dir) 
   export RAYTRACE_RNG_DIR=$(ggeoview-rng-dir) 

   export CUDAWRAP_RNG_MAX=$(ggeoview-rng-max)
} 
ggeoview-run(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   $bin $*
}

ggeoview-lldb()
{
   local bin=$(ggeoview-bin)
   ggeoview-export
   lldb $bin -- $*
}

ggeoview--()
{
    ggeoview-wipe
    ggeoview-cmake
    ggeoview-make
    ggeoview-install
}


ggeoview-depinstall()
{
    bcfg-
    bcfg-install
    bregex-
    bregex-install
    npy-
    npy-install
    ggeo-
    ggeo-install
    assimpwrap- 
    assimpwrap-install
    oglrap-
    oglrap-install
    optixrap-
    optixrap-install 
    thrustrap-
    thrustrap-install 
    ggeoview-
    ggeoview-install  
}


ggeoview-deps-(){ cat << EOD
bcfg
bregex
npy
ggeo
assimpwrap
oglrap
optixrap
cudawrap
thrustrap
EOD
}

ggeoview-deps(){
   local dep
   $FUNCNAME- | while read dep ; do
       $dep-
       #printf "%30s %30s \n" $dep $($dep-idir) 
       echo $($dep-idir)/lib/*.dylib 
   done
}

ggeoview-ls(){   ls -1 $(ggeoview-;ggeoview-deps) ; }
ggeoview-libs(){ otool -L $(ggeoview-;ggeoview-deps) ; }

