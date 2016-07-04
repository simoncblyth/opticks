# === func-gen- : optickscore/optickscore fgp optickscore/optickscore.bash fgn optickscore fgh optickscore
okc-rel(){      echo optickscore ; }
okc-src(){      echo optickscore/optickscore.bash ; }
okc-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(okc-src)} ; }
okc-vi(){       vi $(okc-source) ; }
okc-usage(){ cat << EOU

Brief History
==============


macOS link warning from libOpticksCore.dylib
----------------------------------------------

::

    simon:optickscore blyth$ okc--
    [  0%] Linking CXX shared library libOpticksCore.dylib

    ld: warning: direct access in 
         boost::program_options::typed_value<std::__1::vector<int, std::__1::allocator<int> >, char>::value_type() const 
    to global weak symbol 
        typeinfo for std::__1::vector<int, std::__1::allocator<int> > 
    means the weak symbol cannot be overridden at runtime. 
    This was likely caused by different translation units being compiled with different visibility settings.

    ld: warning: direct access in 
          boost::typeindex::stl_type_index boost::typeindex::stl_type_index::type_id<std::__1::vector<int, std::__1::allocator<int> > >() 
    to global weak symbol 
         typeinfo for std::__1::vector<int, std::__1::allocator<int> > 
    means the weak symbol cannot be overridden at runtime. 
    This was likely caused by different translation units being compiled with different visibility settings.


Only one use of program_options in okc- and one in brap-::

    simon:optickscore blyth$ grep -l program_options *.*
    OpticksCfg.cc

    simon:boostrap blyth$ grep -l program_options *.*
    BCfg.cc
    BCfg.hh

Curiously no such warning from libBoostRap.dylib

Try in BCfg.hh::

     11 #pragma GCC visibility push(default)
     12 #include <boost/program_options.hpp>
     13 #pragma GCC visibility pop

IN okc- try movig CameraCfg implementation from .hh to .cc and 
using explicit instanciation in the .cc.  Results in more warnings.


clang vs clang++ used by boost build ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/19190458/visibility-linker-warnings-when-compiling-ios-app-that-uses-boost?answertab=oldest#tab-top

Some guy: The warnings disappeared after I rebuilt Boost with clang.

* https://github.com/danoli3/ofxiOSBoost/issues/25



2013
------

**Half Year Summary**

Develop G4DAE Geant4 exporter that liberates tesselated G4 geometries
into COLLADA DAE files, including all material and surface properties.

**Aug-Dec 2013**

* study Geant4 and Chroma optical photon propagation
* develop C++ Geant4 geometry exporter : G4DAE 
* experiment with geometry visualizations (webgl, meshlab)

2014 
----------------------------------------------------------------------

**Year Executive Summary**

Get G4DAE exported geometries into Chroma and integrate Geant4 
and Chroma event data via G4DAEChroma runtime bridge.  

**Year Summary**

* Get Chroma to operate with G4DAE exported geometries. 
* Develop G4DAEView visualization using CUDA/OpenGL interoperation techniques
  and OpenGL shaders for geometry and photon visualization.
* Develop G4DAEChroma runtime bridge interfacing Geant4 with external optical photon propagation.
* Realize that photon transport is too large an overhead, so implement GPU Scintillation/Cerenkov
  generation within Chroma based in transported gensteps
   

**Jan-Feb 2014**

* purchase Macbook Pro laptop GPU: NVIDIA GeForce GT 750M
* integrate G4DAE geometry with Chroma 

**Mar-Apr 2014**

* forked Chroma, adding G4DAE integration and efficient interop buffers
* develop g4daeview geometry viewer (based on pyopengl, glumpy)  

**May 2014**

* develop ChromaZMQRoot approach to transporting photons from NuWa to Chroma 

**June/July 2014*

* create GLSL shader visualizations of photon propagations 
* reemission debug 

**August 2014**

* export Daya Bay PMT identifiers
* develop non-graphical propagator

**September 2014**

* present G4DAE geometry exporter at: 19th Geant4 Collaboration Meeting, Okinawa, Sept 2014

**October/November 2014**

* develop G4DAEChroma (photon transport over ZMQ): Geant4 to Chroma runtime bridge 

**December 2014**

* realize photon transport has too much overhead
* implement Cerenkov and Scintillation step transport and photon generation on GPU 

2015
-----

**Year Executive Summary**

Develop Opticks based on the NVIDIA OptiX ray tracing framework, replacing Chroma.
Achieve match between Opticks and Geant4 for simple geometries with speedup 
factor of 200x with a mobile GPU. Performance factor expected to exceed 1000x 
with multi-GPU workstations.  


**Year Summary**

* realize lack of multi-GPU is showstopper for Chroma 
* find that NVIDIA OptiX ray tracing framework exposes accelerated geometry intersection 
* develop Opticks (~15 C++ packages: GGeo, AssimpWrap, OptiXRap, ThrustRap, OGLRap,...) 
  built around NVIDIA OptiX to replace Chroma : effectively 
  recreating part of the Geant4 context on the GPU 
* port Geant4 optical physics into Opticks
* achieve match between Opticks and Geant4 for simple geometries, 
  with speedup factor of 200x with laptop GPU with only 384 cores


**January/February 2015**

* realize lack of multi-GPU support is showstopper for Chroma
* find NVIDIA OptiX, initial tests suggest drastically 50x faster than Chroma
* decide to create Opticks (replacing Chroma) based on NVIDIA OptiX  
* fork Assimp for geometry loading into GGeo model

**March/April 2015**

* workaround OptiX/cuRAND initialization issue by loading persisted cuRAND state into OptiX process 
* create OpenGL visualization package: OGLRap and OptiXEngine ray tracer

**May/June 2015**

* bring NPY persistency to GGeo, use for geocaching
* Cerenkov and Scintillation generated photons match to Geant4 achieved within OptiX machinery
* add GUI to ggeoview using ImGui
* implement Fresnel reflection/refraction with OptiX

**July/August 2015**

* photon indexing using CUDA Thrust 
* OpenGL/OptiX instancing to allow loading of JUNO geometry

**Sept/October 2015**

* finally nail majority of CUDA/Thrust/OpenGL/OptiX interop issues
* photons become GPU resident, using Thrust 
* uncover issue with cleaved meshes, develop fix using OpenMesh
* develop analytic PMT approach 

**November/December 2015** 

* rejig GGeo boundary representation to allow dynamic creation, use for dynamic test geometry
  (ie geometry configurable from commandline)
* create *CfG4* for comparison of test geometries with Geant4
* achieve Geant4/Opticks match with rainbow geometry
* revive compute mode reveals 200x faster performance than Geant4 with only 384 CUDA cores 



Development History
====================

July 2013 (surveying landscape)
-----------------------------------

Looked into muon simulation optimization techniques

* photon weighting

August 2013 (geometry exporter study)
---------------------------------------

* Geant4 Muon simulation profiling, fast-
* studing Geant4 and Geant4/Chroma integration
* looking into Geant4 exporters and visualization
* study meshlab-
* trying VRML exports
* try blender
* study Chroma operation

* https://bitbucket.org/simoncblyth/env/commits/e7cb3c9353775de29bade841b171f7a7682cbe9c

Sept 2013
----------

* sqlite3 based debugging of VRML exports 
* try reality player VRML viewer
* end Sept, start looking into GDML and COLLADA pycollada-
 
Although VRML was a dead end, it provided the G4Polyhedron 
triangulation approach used later in G4DAE.

Sep 24 2013
~~~~~~~~~~~~~

The only real progress so far is with the geometry aspect
where I have made Geant4 exports of VRML2 and GDML
versions of the Dayabay geometry and examined how those
exporters operate. From that experience, I think that
development of a Geant4 Collada exporter (a common 3D file format)
is the most convenient way to proceed in order to
extract the Chroma needed triangles+materials from Geant4.
For developing the new exporter, I need to learn the relevant
parts of the Collada format and can borrow much code
from the VRML2 and GDML exporters.

Oct 2013 (G4DAE approach born)
--------------------------------

* translate Geant4 volume tree into COLLADA DAE
* webpy server of DAE subtrees

November 2013 (G4DAE visualization 1st try: webgl)
----------------------------------------------------

* webgl threejs daeserver.py 

Status report coins G4DAE, were validating G4DAE against VRML2

* https://bitbucket.org/simoncblyth/env/src/9f0c188a8bb2042eb9ad58d95dadf9338e08c634/muon_simulation/nov2013/nov2013_gpu_nuwa.txt?fileviewer=file-view-default

December 2013 (G4DAE visualization 2nd try: meshlab)
-------------------------------------------------------

* meshlab- hijacked for COLLADA viewing
* meshlab COLLADA import terribly slow, and meshlab code is a real mess 
* forked meshlab https://bitbucket.org/simoncblyth/meshlab
* investigate openscenegraph- colladadom- osg-
  (clearly decided meshlab far to messy to be a basis for anything)
 
January 2014
-------------

* purchase Macbook Pro laptop with NVIDIA GeForce GT 750M
* https://bitbucket.org/simoncblyth/env/commits/6da96f0b3617b39bdfeb2bb4f70128335bac23a4
* install CUDA, Chroma : succeed to propagate some photons on GPU

February 2014 (getting Chroma to run)
---------------------------------------

* material and surface properties export into DAE 
* G4DAE collada to chroma conversion

March 2014 (integrating Chroma, geometry visualization 3rd try more fruitful)
--------------------------------------------------------------------------------

* integrate G4DAE with NuWa
* collada to chroma debug
* Chroma ray tracer, quaternion Arcball
* Learning the graphics pipeline https://bitbucket.org/simoncblyth/env/commits/8095bf84c68d99209774af1e47784667351e99f9
* pyopengl- glumpy- yield daeviewgl.py  
* Learning OpenGL/CUDA Interop

April 2014 (getting GPU interop buffers) 
------------------------------------------

* forked Chroma
* standalone-ish render_pbo.py which uses chroma to raycast render to PBO, succeeds to strike geometry, using https://bitbucket.org/scb-/chroma/commits/3a4a443039899af17ca50938d19260cde60bb275
* add orbiting/flyaround mode with DAEParametricView
* rebranding to g4daeview to more accurately reflect the functionality and association with the G4DAE exporter
  https://bitbucket.org/simoncblyth/env/commits/5f5e84fed11fb729e6cefbbcfa07373850005ae5

May 2014 (bring on the photons)
----------------------------------

* ChromaPhotonList for getting photons from NuWa into Chroma
* look into alternative serialization from TObject 
* ChromaPhotonList/NuWa debug 
* ChromaZMQRootTest czrt-
* succeed to connect nuwa.py/geant4(N) with g4daeview.py/chroma(D) via zmq_broker.sh(N)
  https://bitbucket.org/simoncblyth/env/commits/1635c3ffddc2262047a1185a73d5eb6db9c89dec
* ZMQ debugging 

June 2014 (photon visualization, discontent with ancient OpenGL forced by pyopengl)
----------------------------------------------------------------------------------------

* experiment with OpenGL geometry shaders, aiming to avoid vertex doubling up in the photons VBO just to present photon directions as lines
  https://bitbucket.org/simoncblyth/env/commits/ebf0786b0d6f5caa0b7a0cedca7db4c7c4a0a3d2
* photon visualizations using geometry shader 
* migration of photon presentation to GLSL shaders ongoing
* shader integer/bitwise handling remains mysterious, looks like can do almost nothing in glsl 120, maybe workarounf is to do mask selections in CUDA, and communicate results via floats
  https://bitbucket.org/simoncblyth/env/commits/cfc0c38a73cae0f70b08f74343556a358de9b98f
* first cut at VBO propagation, using propagate_vbo.cu in my chroma fork
* resolve shader inflexibility by not trying to delete/recreate shaders but instead keep them around and swap between them

July 2014
-----------

* reemission debug 
* G4DAE explanation targetting non-Physicists : Why HEP Visualization is stuck in the 1990s
  https://bitbucket.org/simoncblyth/env/commits/d0e4dc34b9534946e58c0ce3fe47bc730bdef567

End July/August 2014 (intermission)
---------------------------------------

* migrate to bitbucket 

August 2014
-------------

* Dayabay PMT identifier export
* creating g4daechroma.sh non-graphical ChromaPhotonList responder, that just propagates and replies with no bells/whistles
* looking into alternative numpy persistancy and C/C++ access techniques, cnpy- npyreader-
* prepare presentation for Geant4 Collaboration Meeting
  https://bitbucket.org/simoncblyth/env/commits/7c64db7a621bf07242b22fc47c2f52c75682271e

September 2014
----------------

* present geometry exporter at: 19th Geant4 Collaboration Meeting, Okinawa, Sept 2014
* adjusting to bitbucket 

October 2014
--------------

* brief look into swift, scenekit for checking G4DAE exports
* g4daescenekit.swift script to check SceneKit parsing of G4DAE exports from commandline 
* add extra SkinSurface and OpticalSurface objects to DAE level geometry in order to be transformed into sensitive surfaces needed for chroma SURFACE_DETECT
* G4DAEChroma : Geant4 to external bridge

November 2014
----------------
 
* G4DAEChroma C++ bridge : fleshout, debug

December 2014
---------------

* start attempt to parallelize Cerenkov and Scintillation photon generation, by persisting stacks from just before big photon generation for loops
* debug Scintillation/Cerenkov

January 2015
-------------

* https://bitbucket.org/simoncblyth/env/src/2373bb7245ca3c1b8fb06718d4add402805eab93/presentation/gpu_accelerated_geant4_simulation.txt?fileviewer=file-view-default

  * G4 Geometry model implications 
  * G4DAE Geometry Exporter
  * G4DAEChroma bridge

* first look at OptiX immediately after making the above presentation
* investigate Assimp for geometry loading 
* succeed to strike geometry with Assimp and OptiX

February 2015
----------------

* fork Assimp https://github.com/simoncblyth/assimp/commits/master
* benchmarks with using CUDA_VISIBLE_DEVICES to control how many K20m GPUs are used
* fork Assimp for Opticks geometry loading
* test OptiX scaling with IHEP GPU machine
* great GGeo package, intermediary geometry model
* experiment with GPU textures for interpolated material property access 

March 2015
-----------

* encounter OptiX/cuRAND resource issue, workaround using pure CUDA to initialize and persist state
* fail to find suitable C++ higher level OpenGL package, start own oglrap- on top of GLFW, GLEW
* integrate ZMQ messaging with NPY serialization using Boost.ASIO ASIO-ZMQ to create NumpyServer

April 2015
------------

* reuse NumpyServer infrastructure for UDP messaging allowing live reconfig of objects 
  with boost::program_option text parsing 
* add quaternion Trackball for interactive control
* avoid duplication with OptiXRap
* arrange OptiX output buffer to be a PBO which is rendered as texture by OpenGL
* create OpenGL Prog/Shadr infrastructure in oglrap-  
* OptiXEngine starting point for propagation, previously focussed on OptiX ray tracing 
* ported Cerenkov generation from Chroma to OptiX

May 2015
----------

* bring NPY persistency to GGeo
* implement geocache loading to avoid XML parsing on every launch 
  (turned out to be a luxury for DayaBay [saving only a few seconds per launch], 
   but 6 months hence it is a necessity for JUNO [saving several minutes for every launch])
* GSubstanceLib infrastructure
* reemission handling, inverse CDF texture creation
* Cerenkov and Scintillation generated photons match to Geant4 achieved within OptiX machinery
* pick ImGui immediate mode GUI renderer
* GUI adoption by the oglrap classes
* prepare presentation 

  * Why not Chroma ? Progress report on migrating to OptiX 
  * http://simoncblyth.bitbucket.org/env/presentation/optical_photon_simulation_with_nvidia_optix.html

June 2015
-----------

* efficient Fresnel using CG computation techniques
* photon flag handling 
* photon record compression
* implement animation
* investigate image space photon mapping ispm-
* photon indexing using CPU
* learning CUDA Thrust 
* update OptiX to 3.8.0 and start interop from OptiX to Thrust 1.8 that comes with CUDA 7.0

July 2015
------------

* create thrustrap- to hold indexing code
* CUDA/Thrust/OpenGL/OptiX interop debugging
* instancing : start implementation
* modifications to get JUNO geometry to load
* get OpenGL instancing operational

August 2015
------------

* OptiX instancing 
* introduce BBox rendering for speed
* JUNO genstep debug 

Sept 2015
-----------

* attacking the CUDA/Thrust/OpenGL/OptiX interop issues
* demo code for stream compaction operational OptiXThrust::compaction
  https://bitbucket.org/simoncblyth/env/commits/398811a731ffc4caef3f07fdc18362b842d98c37 
* succeed to seed the OptiX photon buffer with genstep indices entirely on GPU, using Thrust based OBuf OBufPair
* add pickphoton live option for pointing view at a single photon path and optionally hiding others
* add TORCH genstep, a more controllable source of photons
* add wireframe to Scene GeometryStyles, to see the mesh : detdesc spelunking IAV, 
  looks like geometry issue is a split union solid maybe due to polycons with coincident z planes
* add pickface option for selecting single or ranges of faces, use to identify internal faces of the split union solid 3158
* learn OpenMesh, develop mesh surgery in openmeshrap-

October 2015
-------------

* App::checkGeometry reveals the extent of the topological problems, about 10% of dyb meshes with issues,
  fortunately only a few critical must fix ones
* instanced identity handling 
* analytic PMT description
* OptiX analytic intersection code
* split up OEngine monolith
* Camera rejig that brings OpenGL orthographic and perspective projections closer and matches with OptiX raygen equivalents
* dynamic boundary buffer creation with GBndLib
* ggeo- refactoring 

November 2015
--------------

* GTestBox making use of GBndLib dynamic boundary creation
* rejig GMergedMesh in preparation for dynamic combination of GSolids with a GMergedMesh
* Fresnel reflection S/P polz debug
* tidy reflection check, moving focus away from triangulation cracks yields near perfect Fresnel S/P polarization reflection curve agreement
* plan a more appropriate volume to surface translation of analytic PMT geometry
* extract triangle subdivision functionality from icosahedron.cpp into NTesselate for application to other basis polyhedrons
* geodesic subdiv of hemi octahedron creates tesselated hemi-sphere
* prism deviation angle check
* parse refractive index csv from http://refractiveindex.info into numpy arrays
* investigate CIE XYZ determination from spectra using color matching functions and convertion into RGB color spaces

December 2015
----------------

* numpy generation of wavelengths according to Planck 6500K blackbody formula, using binned cdf inversion
* wavelength domain debug and rendering simulated 3M photon rainbows with ggv-rainbow and npy-/rainbow.py
* cfg4 : new package for comparison against standalone geant4
* workaround G4 limitation in number of primaries by splitting photon squadron across multiple events
* creating opticks format photon/step/history records with cfg4-
* try to duplicate G4 approach to polarization at boundaries in propagate_at_boundary_geant4_style
* move seqhis and seqmat indices to NumpyEvt and persist their, allowing use of the indices with loaded NumpyEvt
* enable loading of photons/records into ggv, allowing visualization of both Opticks and G4 cfg4- generated/propagated events
* adjust cfg4- truncation to match Opticks, get chi2 match with rainbow_cfg4.py

January 2016
--------------

* revive compute mode
* split validation/compilation/prelaunch from launch timings, gets 1M rainbow down to 0.29s in compute mode
* try a new tack on State control aiming to replace or drastically change the long broken Bookmarks
* rework Bookmarks to handle just jumping between states, the state persisting and applying being done by NState
* use State machinery to implement animated interpolated View, ie fly around geometry 




Related
========


* http://www.nvidia.com/content/GTC/posters/40_AbuZayyad_Monte_Carlo_Simulation.pdf


EOU
}

okc-env(){      olocal- ; opticks- ;  }

okc-sdir(){ echo $(opticks-home)/optickscore ; }
okc-tdir(){ echo $(opticks-home)/optickscore/tests ; }
okc-idir(){ echo $(opticks-idir) ; }
okc-bdir(){ echo $(opticks-bdir)/$(okc-rel) ; }

okc-scd(){  cd $(okc-sdir); }
okc-tcd(){  cd $(okc-sdir); }
okc-cd(){   cd $(okc-sdir)/$1 ; }
okc-icd(){  cd $(okc-idir); }
okc-bcd(){  cd $(okc-bdir); }

okc-bin(){ echo $(okc-idir)/bin/${1:-OpticksResourceTest} ; }

okc-wipe(){ local bdir=$(okc-bdir) ; rm -rf $bdir ;  }

okc-name(){ echo OpticksCore ; }
okc-tag(){  echo OKCORE ; }

okc--(){        opticks--     $(okc-bdir) ; }
okc-ctest(){    opticks-ctest $(okc-bdir) $* ; }
okc-genproj(){  okc-scd ; opticks-genproj $(okc-name) $(okc-tag) ; }
okc-gentest(){  okc-tcd ; opticks-gentest ${1:-OpticksGeometry} $(okc-tag) ; }
okc-txt(){ vi $(okc-sdir)/CMakeLists.txt $(okc-tdir)/CMakeLists.txt ; }




