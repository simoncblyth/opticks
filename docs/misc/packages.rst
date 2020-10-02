Opticks Packages Overview
===========================

Opticks is structured as a collection of ~20 modular projects 
organized by their dependencies. This structure allows Opticks
to be useful is different circumstances, for example on machines without 
an NVIDIA GPU capable of running CUDA and OptiX the OpenGL visualization 
functionality can still be used.

This structure also provides fast rebuilding as typically during 
development it is only necessary to rebuild one or two packages.


Package Dependencies
----------------------

=====================  ===============  ===============   ==============================================================================
directory              precursor        pkg name          required find packages 
=====================  ===============  ===============   ==============================================================================
sysrap                 sysrap-          SysRap            PLog
boostrap               brap-            BoostRap          OpticksBoost PLog SysRap
opticksnpy             npy-             NPY               OpticksBoost PLog SysRap BoostRap GLM
optickscore            okc-             OpticksCore       OpticksBoost PLog SysRap BoostRap GLM NPY 
ggeo                   ggeo-            GGeo              OpticksBoost PLog SysRap BoostRap GLM NPY OpticksCore (abbreviation:BASE)
assimprap              assimprap-       AssimpRap         BASE GGeo Assimp
openmeshrap            openmeshrap-     OpenMeshRap       BASE GGeo OpenMesh
opticksgeo             okg-             OpticksGeometry   BASE GGeo Assimp AssimpRap OpenMesh OpenMeshRap      
oglrap                 oglrap-          OGLRap            BASE GGeo GLEW GLFW ImGui        
cudarap                cudarap-         CUDARap           PLog SysRap CUDA (ssl) 
thrustrap              thrustrap-       ThrustRap         OpticksBoost PLog SysRap BoostRap GLM NPY CUDA CUDARap 
optixrap               oxrap-           OptiXRap          BASE GGeo Assimp AssimpRap CUDARap ThrustRap
okop                   okop-            OKOP              BASE GGeo OptiX OptiXRap CUDA CUDARap ThrustRap OpticksGeometry     
opticksgl              opticksgl-       OpticksGL         BASE GGeo OptiX OptiXRap CUDA CUDARap ThrustRap OpticksOp Assimp AssimpRap GLEW GLFW OGLRap 
ok                     ok-              OK                BASE GGeo Assimp AssimpRap OpenMesh OpenMeshRap OpticksGeometry GLEW GLFW ImGui OGLRap 
cfg4                   cfg4-            cfg4              BASE GGeo Geant4 EnvXercesC [G4DAE] 
okg4                   okg4-            okg4              BASE GGeo Assimp AssimpRap OpenMesh OpenMeshRap OpticksGeometry GLEW GLFW ImGui OGLRap Geant4 EnvXercesC
=====================  ===============  ===============   ==============================================================================




Roles of the Opticks projects
---------------------------------

okconf
    detects versions of Geant4, OptiX available 
sysrap
    logging, string handling, envvar handling 
boostrap
    filesystem utils, regular expression matching, commandline parsing 
npy
    array handling, persistency in NumPy format, mesh handling/polygonization, nnode, NCSG
    (currently this package is too big, it needs to be split)
optickscore
    definitions, loosely the model of the app 
yoctoglrap
    wrapper for the YoctoGL external, providing glTF 2.0 3D file format parsing/writing
ggeo
    geometry representation appropriate for uploading to the GPU
assimprap
    wrapper for Assimp 3D geometry importer, can load G4DAE COLLADA geometry files
    (no longer needed with ExtG4 direct from Geant4 conversion)
openmeshrap
    wrapper for OpenMesh, providing mesh traversal : used for mesh fixing 
    (no longer needed with analytic geometry) 
opticksgeo
    bring together ggeo, assimprap and openmeshrap to load and fix geometry
oglrap
    wrapper for OpenGL, visualization of geometry and photon propagations :
    OpenGL rendering, including GLSL shader sources
cudarap
    loading curand persisted state
thrustrap
    fast GPU photon indexing using interop techniques 
optixrap
    conversion of GGeo geometry into OptiX GPU geometry, OptiX programs for propagation 
okop
    pure compute propagation, with no OpenGL dependency, operations, high level OptiX control 
opticksgl 
    OpenGL/OptiX/CUDA interop propagation using shared OpenGL buffers, 
    allowing visualization of propagations directly from the shared interop
    GPU buffers 
ok
    high level OKMgr and OKPropagator, pulling together all the above
extg4
    translates Geant4 solids into NCSG trees and Geant4 trees of volumes
    into GGeo geometries 
cfg4
    contained geant4, comparison of Geant4 and Opticks simulations
okg4
    full integration of Opticks and Geant4 including:

    * Geant4 non-optical simulation (and optical too whilst testing)
    * Geant4 GDML detector geometry loading 
    * Opticks DAE geometry loading etc...
    * optixrap: OptiX optical propagation
    * oglrap: OpenGL visualization
    * thrustrap: Thrust GPU indexing 
g4ok
    top level (non-visualization) project intended to provide simple 
    interface between Geant4 code with embedded Opticks : to be 
    used from Geant4 examples


Geant4 Dependency
-------------------

Only a few of the very highest level packages depend on Geant4. 

extg4
     geometry translation
cfg4
     validation comparisons
okg4
     integrated Opticks+G4 for “gun running"
g4ok
     minimal interface for embedding Opticks inside Geant4 applications


Opticks dependency on Geant4 is intended to be loose 
in order to allow working with multiple G4 versions (within a certain version range), 
using version preprocessor macros to accommodate differences.  
So please send copy/paste reports of incompatibilities together with G4 versions.

The weak G4 dependency allows testing of much of Opticks even 
without G4 installed.  




