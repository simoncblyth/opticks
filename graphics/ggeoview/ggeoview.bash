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


issue: jpmt timeouts binary search to pin down 
------------------------------------------------

::

    ggv --juno 
       # no pmt, evt propagation vis not working 

    ggv --jpmt --modulo 1000 
       # causes a timeout+freeze requiring a reboot

    ggv --jpmt --modulo 1000 --nopropagate
       # can visualize jpmt OptiX geometry: and it looks OK

    ggv --jpmt --modulo 1000 --trivial
       # swap generate program with a trivial standin  : works 

    ggv --jpmt --modulo 1000 --bouncemax 0
       # just generate, no propagation : timeout+freeze, reboot
       
    ggv --jpmt --modulo 1000 --trivial
       # progressively adding lines from generate into trivial
       # suggests first issue inside generate_cerenkov_photon/wavelength_lookup

    ggv --jpmt --modulo 1000 --bouncemax 0
       # just generate, no propagation : works after kludging wavelength_lookup 
       # to always give a constant valid float4

    ggv --jpmt --modulo 100
       #  still with kludged wavelength_lookup : works, with photon animation operational

    ggv --jpmt --modulo 50
       #  still with kludged wavelength_lookup : timeout...  maybe stepping off reservation somewhere else reemission texture ?


issue: jpmt wavelengthBuffer/boundarylib ? maybe bad material indices ?
-------------------------------------------------------------------------

* is the cs.MaterialIndex expected to be the wavelength texture line number ?

  * if so then the jpmt/juno numbers do need a "translation" applied ?
  * GBoundaryLibMetadata.json has 18 boundaries 0..17

::

    In [5]: cd /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae
    /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae

    In [6]: a = np.load("wavelength.npy")

    In [40]: a.reshape(-1,6,39,4).shape
    Out[40]: (18, 6, 39, 4)

    In [47]: a.reshape(-1,6,39,4)[6]
    Out[47]: 
    array([[[       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.345,      273.208,  1000000.   ,        0.   ],
            [       1.36 ,      273.208,  1000000.   ,        0.   ],
            [       1.375,      273.208,  1000000.   ,        0.   ],
            [       1.39 ,      691.558,  1000000.   ,        0.   ],
            [       1.384,     1507.119,  1000000.   ,        0.   ],

    In [54]: 18*6
    Out[54]: 108

::

    delta:npy blyth$ /usr/local/env/numerics/npy/bin/NPYTest
    [2015-08-28 21:14:53.244421] [0x000007fff7650e31] [debug]   NPY<T>::load /usr/local/env/juno/cerenkov/1.npy
    G4StepNPY
     ni 3840 nj 6 nk 4 nj*nk 24 
     (    0,    0)               -1                1               48              322  sid/parentId/materialIndex/numPhotons 
     (    0,    1)            0.000            0.000            0.000            0.000  position/time 
     (    0,    2)           -0.861           -0.156           -0.530            1.023  deltaPosition/stepLength 
     (    0,    3)               13           -1.000            1.000          299.792  code 
     (    0,    4)            1.000            0.000            0.000            0.688 
     (    0,    5)            0.527          293.245          293.245            0.000 
     ( 3839,    0)           -38391                4               48               47  sid/parentId/materialIndex/numPhotons 
     ( 3839,    1)          -16.246           -2.947          -10.006            0.064  position/time 
     ( 3839,    2)           -0.191           -0.194            0.236            0.378  deltaPosition/stepLength 
     ( 3839,    3)               11           -1.000            1.000          230.542  code 
     ( 3839,    4)            1.300            0.000            0.000            0.895 
     ( 3839,    5)            0.200          165.673          110.064            0.000 
     24 
     42 
     48 
     24 : 750 
     42 : 52 
     48 : 3038 


* TODO : trace the prop values



issue: ~/jpmt_mm0_too_many_vertices.txt
------------------------------------------

1.79M vertices for jpmt mm0 (global) seems excessive, either missing a repeater or some bug.::

    ggv -G --jpmt

    120 [2015-Aug-25 18:52:37.665158]: GMergedMesh::create index 0 from default root base lWorld0x22ccd90
    121 [2015-Aug-25 18:52:37.730168]: GMergedMesh::create index 0 numVertices 1796042 numFaces 986316 numSolids 289733 numSolidsSelected 1032


From m_mesh_usage in GGeo and GMergedMesh sStrut and sFasteners are the culprits::

    [2015-Aug-25 19:50:40.251333]: AssimpGGeo::convertMeshes  i   19 v  312 f  192 n sStrut0x304f210
    [2015-Aug-25 19:50:40.251575]: AssimpGGeo::convertMeshes  i   20 v 3416 f 1856 n sFasteners0x3074ea0

    [2015-Aug-25 19:54:01.663594]: GMergedMesh::create index 0 from default root base lWorld0x22ccd90
    [2015-Aug-25 19:54:07.339150]: GMergedMesh::create index 0 numVertices 1796042 numFaces 986316 numSolids 289733 numSolidsSelected 1032
    GLoader::load reportMeshUsage (global)
         5 :     62 : sWall0x309ce60 
         6 :      1 : sAirTT0x309cbb0 
         7 :      1 : sExpHall0x22cdb00 
         8 :      1 : sTopRock0x22cd500 
         9 :      1 : sTarget0x22cfbd0 
        10 :      1 : sAcrylic0x22cf9a0 
        19 :    480 : sStrut0x304f210 
        20 :    480 : sFasteners0x3074ea0 
        21 :      1 : sInnerWater0x22cf770 
        22 :      1 : sReflectorInCD0x22cf540 
        23 :      1 : sOuterWaterPool0x22cef90 
        24 :      1 : sSteelTub0x22ce610 
        25 :      1 : sBottomRock0x22cde40 
            ---------

    In [7]: 480+480+62+10
    Out[7]: 1032          ## matches numSolidsSelected

    In [5]: 3416*480+312*480
    Out[5]: 1789440


::

    simon:juno blyth$ grep sFasteners t3.dae
        <geometry id="sFasteners0x3074ea0" name="sFasteners0x3074ea0">
            <source id="sFasteners0x3074ea0-Pos">
              <float_array count="2742" id="sFasteners0x3074ea0-Pos-array">
                <accessor count="914" source="#sFasteners0x3074ea0-Pos-array" stride="3">
            <source id="sFasteners0x3074ea0-Norm">
              <float_array count="5184" id="sFasteners0x3074ea0-Norm-array">
                <accessor count="1728" source="#sFasteners0x3074ea0-Norm-array" stride="3">
            <source id="sFasteners0x3074ea0-Tex">
              <float_array count="2" id="sFasteners0x3074ea0-Tex-array">
                <accessor count="1" source="#sFasteners0x3074ea0-Tex-array" stride="2">
            <vertices id="sFasteners0x3074ea0-Vtx">
              <input semantic="POSITION" source="#sFasteners0x3074ea0-Pos"/>
              <input offset="0" semantic="VERTEX" source="#sFasteners0x3074ea0-Vtx"/>
              <input offset="1" semantic="NORMAL" source="#sFasteners0x3074ea0-Norm"/>
              <input offset="2" semantic="TEXCOORD" source="#sFasteners0x3074ea0-Tex"/>
              <meta id="sFasteners0x3074ea0">
          <instance_geometry url="#sFasteners0x3074ea0">
    simon:juno blyth$ 



Contiguous block of Fasteners all leaves at depth 6::

    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | wc -l
         480
    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | head -1
       282429 [  6:54799/55279]    0          lFasteners0x3075090   
    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | tail -1
       282908 [  6:55278/55279]    0          lFasteners0x3075090   

    In [9]: 282429+480-1
    Out[9]: 282908


TODO: 


* on dyb GTreeCheck::findRepeatCandidates 

  * not restricting repeats to non-leaf looses some geometry
  * but putting it back gives PMTs in both instance0 and instance1  
  * GTreeCheck.dumpTree ridx not making sense when allow leaf repeats 

* dump the text node tree for juno, to see where sFasteners is 
* add --repeatidx 0,1,2,3 controlled loading in GGeo::loadMergedMeshes etc..
  so can skip the problematic extremely large 0




squeeze approaches for jpmt
----------------------------

* remove vertex color, do at solid/boundary level
* compress vertex normals 
* reuse vertex structures for OptiX ?



computeTest with different core counts controlled via CUDA_VISIBLE_DEVICES
----------------------------------------------------------------------------

Juno Scintillation 2, genstep scaledown 25
--------------------------------------------

::

    genstepAsLoaded : 4e16b039dc40737a4c0c51d7b213a118
    genstepAfterLookup : 4e16b039dc40737a4c0c51d7b213a118
               Type :   scintillation
                Tag :               1
           Detector :            juno
        NumGensteps :            1774
             RngMax :         3000000
         NumPhotons :         1493444
         NumRecords :        14934440
          BounceMax :               9
          RecordMax :              10
        RepeatIndex :              10
         photonData : 33b5c1f991b46e09036e38c110e36102
         recordData : 55a15aacf09d4e8dcf269d6e882b481e
       sequenceData : 035310267fc2a678f2c8cad2031d7101




::

    2.516              GT 750M          ggv.sh --cmp --juno -s 
 
    0.487              GTX 750 Ti 


    0.153      -         Tesla K40m  ( 11520 )

    0.157      0,1,2,3 

                      
              Tesla K40m   (5760)

    0.201      0,1                      
    0.200      2,3                      

    0.179      1,2                     
    0.179      0,2                      
    0.178      1,3                      
     
    0.202      0,1,2
    0.201      0,1,3

    0.134      1,2,3
 


::

    In [1]: 2.516/0.134
    Out[1]: 18.776119402985074




Juno Cerenkov 1, scaledown ?10
---------------------------------

::

    0.126,0.126   0         Tesla K40m  2880 CUDA cores  
    0.127         1
    0.127         2
    0.126         3
  
    0.088,0.087   0,1             5760 
    0.076         0,2
    0.099         2,3
    0.080         1,3

    0.076         0,1,2           8640
    0.058         1,2,3
    0.057         1,2,3

    0.062         0,1,2,3         11520
    0.062,0.062,0.062,0.063   NO ENVVAR
    

    1.130          GT750M    ggv.sh --juno --cmp      384 CUDA cores
    1.143 
    1.146 
    1.137 
    1.139 


    0.195,0.197    GTX 750 Ti    640 CUDA Cores                             


    a = np.array( [[384, 1.130],[640,0.195],[2880,0.126],[5760,0.080],[8640,0.070],[11520,0.062]] )

    plt.plot( a[:,0], a[0,-1]/a[:,1], "*-")








GGeoview Compute 
------------------

Compute only mode::

   ggeoview-compute -b0
   ggeoview-compute -b1
   ggeoview-compute -b2
   ggeoview-compute -b4   # up to 4 bounces working 

   ggeoview-compute -b5   # crash for beyond 4  



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
ggeoview-gdir(){ echo $(ggeoview-idir).generated ; }

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
ggeoview-compute-name(){ echo computeTest ; }

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

ggeoview-compute-bin(){ echo $(ggeoview-idir)/bin/$(ggeoview-compute-name) ; }

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

ggeoview-detector()
{
    echo ${GGEOVIEW_DETECTOR:-DAE_NAME_DYB}
}




ggeoview-detector-juno()
{
    export GGEOVIEW_DETECTOR=DAE_NAME_JUNO
}
ggeoview-detector-dyb()
{
    export GGEOVIEW_DETECTOR=DAE_NAME_DYB
}


ggeoview-query-dyb() {
    echo range:3153:12221
   #q="range:3153:4814"     #  transition to 2 AD happens at 4814 
   #q="range:3153:4813"     #  this range constitutes full single AD
   #q="range:3161:4813"      #  push up the start to get rid of plain outer volumes, cutaway view: udp.py --eye 1.5,0,1.5 --look 0,0,0 --near 5000
   #q="index:5000"
   #q="index:3153,depth:25"
   #q="range:5000:8000"
}

ggeoview-query-juno() {
    #echo range:1:289733
    #echo range:1:100000   # OpenGL vis works but slowly
    echo range:1:50000    # 
}
ggeoview-query-jpmt() {
    echo range:1:289734    #   289733+1 all test3.dae volumes 
    #echo range:1:50000    # 
}
ggeoview-query-jtst() {
    echo range:1:50000    
}


ggeoview-query() {
    if [ "$(ggeoview-detector)" == "DAE_NAME_DYB" ]; then
        ggeoview-query-dyb
    elif [ "$(ggeoview-detector)" == "DAE_NAME_JUNO" ]; then
        ggeoview-query-juno
    elif [ "$(ggeoview-detector)" == "DAE_NAME_JPMT" ]; then
        ggeoview-query-jpmt
    elif [ "$(ggeoview-detector)" == "DAE_NAME_JTST" ]; then
        ggeoview-query-jtst
    fi
}

# TODO: find cleaner way, detector specifics dont belong here...







ggeoview-export()
{
   export-
   export-export

   export GGEOVIEW_GEOKEY="$(ggeoview-detector)"
   export GGEOVIEW_QUERY="$(ggeoview-query)"
   export GGEOVIEW_CTRL=""

   export SHADER_DIR=$(ggeoview-sdir)/gl
   export SHADER_DYNAMIC_DIR=$(ggeoview-gdir)
   export SHADER_INCL_PATH=$(ggeoview-sdir)/gl:$(ggeoview-sdir):$SHADER_DYNAMIC_DIR
   mkdir -p $SHADER_DYNAMIC_DIR

   export RAYTRACE_PTX_DIR=$(ggeoview-ptx-dir) 
   export RAYTRACE_RNG_DIR=$(ggeoview-rng-dir) 

   export CUDAWRAP_RNG_MAX=$(ggeoview-rng-max)
} 

ggeoview-export-dump()
{
   env | grep GGEOVIEW
   env | grep SHADER
   env | grep RAYTRACE
   env | grep CUDAWRAP

}

ggeoview-run(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   $bin $*
}

ggeoview-compute(){ 
   local bin=$(ggeoview-compute-bin)
   ggeoview-export
   $bin $*
}

ggeoview-compute-lldb(){ 
   local bin=$(ggeoview-compute-bin)
   ggeoview-export
   lldb $bin $*
}

ggeoview-compute-gdb(){ 
   local bin=$(ggeoview-compute-bin)
   ggeoview-export
   gdb --args $bin $*
}



ggeoview-vrun(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   vglrun $bin $*
}

ggeoview-gdb(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   gdb --args $bin $*
}

ggeoview-valgrind(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   valgrind $bin $*
}

ggeoview-lldb()
{
   local bin=$(ggeoview-bin)
   ggeoview-export
   lldb $bin -- $*
}

ggeoview-dbg()
{
   case $(uname) in
     Darwin) ggeoview-lldb $* ;;
          *) ggeoview-gdb  $* ;;
   esac
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

ggeoview-depcmake()
{
   local dep
   ggeoview-deps- | while read dep ; do
       $dep-
       $dep-cmake
   done
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
   local suffix=${1:-dylib}
   local dep
   $FUNCNAME- | while read dep ; do
       $dep-
       #printf "%30s %30s \n" $dep $($dep-idir) 
       echo $($dep-idir)/lib/*.${suffix}
   done
}

ggeoview-ls(){   ls -1 $(ggeoview-;ggeoview-deps) ; }
ggeoview-libs(){ otool -L $(ggeoview-;ggeoview-deps) ; }

ggeoview-linux-setup() {
    local dep
    local edeps="boost glew glfw imgui glm assimp"
    local deps="$edeps bcfg bregex npy ggeo assimpwrap ppm oglrap cudawrap optixrap thrustrap"
    for dep in $deps
    do
        $dep-
        [ -d "$($dep-idir)/lib" ] &&  export LD_LIBRARY_PATH=$($dep-idir)/lib:$LD_LIBRARY_PATH
        [ -d "$($dep-idir)/lib64" ] &&  export LD_LIBRARY_PATH=$($dep-idir)/lib64:$LD_LIBRARY_PATH
    done

    assimp-
    export LD_LIBRARY_PATH=$(assimp-prefix)/lib:$LD_LIBRARY_PATH
}

ggeoview-linux-install-external() {
    local edeps="glew glfw imgui glm assimp"
    local edep
    for edep in $edeps
    do
        ${edep}-
        ${edep}-get
        ${edep}--
    done
}
ggeoview-linux-install() {

    local deps="bcfg bregex npy ggeo assimpwrap ppm oglrap cudawrap optixrap thrustrap"
    local dep

    for dep in $deps
    do
        $dep-
        $dep--
    done
}
