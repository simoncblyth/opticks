Opticks Overview
==================

Opticks is structured as a collection of modular projects 
organized by their dependencies. This structure allows Opticks
to be useful is different circumstances, for example on machines without 
an NVIDIA GPU capable of running CUDA and OptiX the OpenGL visualization 
functionality can still be used.


Project Dependencies
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
opticksgeo             opticksgeo-      OpticksGeometry   BASE GGeo Assimp AssimpRap OpenMesh OpenMeshRap      
oglrap                 oglrap-          OGLRap            BASE GGeo GLEW GLFW ImGui        
cudarap                cudarap-         CUDARap           PLog SysRap CUDA (ssl) 
thrustrap              thrustrap-       ThrustRap         OpticksBoost PLog SysRap BoostRap GLM NPY CUDA CUDARap 
optixrap               optixrap-        OptiXRap          BASE GGeo Assimp AssimpRap CUDARap ThrustRap
opticksop              opticksop-       OpticksOp         BASE GGeo OptiX OptiXRap CUDA CUDARap ThrustRap      
opticksgl              opticksgl-       OpticksGL         BASE GGeo OptiX OptiXRap CUDA CUDARap ThrustRap OpticksOp Assimp AssimpRap GLEW GLFW OGLRap 
ggeoview               ggeoview-        GGeoView          BASE GGeo Assimp AssimpRap OpenMesh OpenMeshRap OpticksGeometry GLEW GLFW ImGui OGLRap 
cfg4                   cfg4-            CfG4              BASE GGeo Geant4 EnvXercesC [G4DAE] 
okg4                   okg4-            okg4              BASE GGeo Assimp AssimpRap OpenMesh OpenMeshRap OpticksGeometry GLEW GLFW ImGui OGLRap Geant4 EnvXercesC
=====================  ===============  ===============   ==============================================================================


Roles of the Opticks projects
---------------------------------

sysrap
    logging, string handling, envvar handling 
boostrap
    filesystem utils, regular expression matching, commandline parsing 
opticksnpy
    array handling 
optickscore
    definitions, loosely the model of the app 
ggeo
    geometry representation appropriate for uploading to the GPU
assimprap
    parsing G4DAE geometry file into the GGeo representation  
openmeshrap
    geometry fixing
opticksgeo
    bring together ggeo, assimprap and openmeshrap to load and fix geometry
oglrap
    OpenGL rendering, including GLSL shader sources
cudarap
    loading curand persisted state
thrustrap
    fast GPU photon indexing using interop techniques 
optixrap
    conversion of GGeo geometry into OptiX GPU geometry, OptiX programs for propagation 
opticksop
    high level OptiX control 
opticksgl 
    combination of oglrap- OpenGL and OptiX raytracing 
ggeoview
    putting together all the above
cfg4
    contained geant4, comparison of Geant4 and Opticks simulatons




