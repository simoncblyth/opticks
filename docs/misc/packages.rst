Opticks Packages Overview
===========================

Opticks is structured as a collection of ~20 modular projects 
organized by their dependencies. This structure allows Opticks
to be useful is different circumstances, for example on machines without 
an NVIDIA GPU capable of running CUDA and OptiX the OpenGL visualization 
functionality can still be used.

This structure also provides fast rebuilding as typically during 
development it is only necessary to rebuild one or two packages.


Show Package Dependencies with opticks-deps
-----------------------------------------------

opticks-deps parses CMakeList.txt files to discern dependencies between packages::


    epsilon:opticks blyth$ opticks-deps
    [2021-01-14 18:57:39,007] p90686 {/Users/blyth/opticks/bin/CMakeLists.py:165} INFO - home /Users/blyth/opticks 
              API_TAG :        reldir :         bash- :     Proj.name : dep Proj.names  
     10        OKCONF :        okconf :        okconf :        OKConf : OpticksCUDA OptiX G4  
     20        SYSRAP :        sysrap :        sysrap :        SysRap : OKConf PLog  
     30          BRAP :      boostrap :          brap :      BoostRap : Boost BoostAsio NLJSON PLog SysRap Threads  
     40           NPY :           npy :           npy :           NPY : PLog GLM BoostRap  
     50        OKCORE :   optickscore :           okc :   OpticksCore : NPY  
     60          GGEO :          ggeo :          ggeo :          GGeo : OpticksCore  
     90         OKGEO :    opticksgeo :           okg :    OpticksGeo : OpticksCore GGeo  
    100       CUDARAP :       cudarap :       cudarap :       CUDARap : SysRap OpticksCUDA  
    110         THRAP :     thrustrap :         thrap :     ThrustRap : OpticksCore CUDARap  
    120         OXRAP :      optixrap :         oxrap :      OptiXRap : OKConf OptiX OpticksGeo ThrustRap  
    130          OKOP :          okop :          okop :          OKOP : OptiXRap  
    140        OGLRAP :        oglrap :        oglrap :        OGLRap : ImGui OpticksGLEW BoostAsio OpticksGLFW OpticksGeo  
    150          OKGL :     opticksgl :          okgl :     OpticksGL : OGLRap OKOP  
    160            OK :            ok :            ok :            OK : OpticksGL  
    165            X4 :         extg4 :            x4 :         ExtG4 : G4 GGeo OpticksXercesC CLHEP  
    170          CFG4 :          cfg4 :          cfg4 :          CFG4 : G4 ExtG4 OpticksXercesC OpticksGeo ThrustRap  
    180          OKG4 :          okg4 :          okg4 :          OKG4 : OK CFG4  
    190          G4OK :          g4ok :          g4ok :          G4OK : CFG4 ExtG4 OKOP  
    200          None :   integration :   integration :   Integration :   



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
ggeo
    geometry representation appropriate for uploading to the GPU
opticksgeo
    following removals of assimprap and openmeshrap this has started to 
    become vestigial middle management 
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


Roles of former packages
----------------------------

assimprap
    wrapper for Assimp 3D geometry importer, can load G4DAE COLLADA geometry files
    (no longer needed with ExtG4 direct from Geant4 conversion)
openmeshrap
    wrapper for OpenMesh, providing mesh traversal : used for mesh fixing 
    (no longer needed with analytic geometry) 
yoctoglrap
    wrapper for the YoctoGL external, providing glTF 2.0 3D file format parsing/writing



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




