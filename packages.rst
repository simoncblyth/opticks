Opticks Packages
==================

Base
------

okconf
    config version detection
sysrap
    basis types, new array NP.hh
boostrap
    boost tools
npy
    geo primitives, old array NPY.hpp
optickscore
    old core, argument parsing  

Geometry 
----------------------------

ggeo
    complete geometry model : no Geant4 dependency
extg4
    Geant4 geometry translation into GGeo model 
opticksgeo
    high level pkg on top of ggeo
cfg4
    Old Geant4 comparison machinery, eg CRecorder 


Old CUDA Simulation and Visualization
-----------------------------------------

cudarap
    old CUDA interface
thrustrap
    old CUDA thrust interface : photon seeding, indexing 
optixrap
    old optical photon simulation implemented in old OptiX API 
okop
    high level pkg on top of optixrap 
oglrap
    old OpenGL based visualization of opticksgeo geometry
opticksgl
    integrating OpenGL with okop OptiX 


New Geometry model and CUDA simulation  
----------------------------------------------------

CSG_GGeo
    GGeo to CSG geometry translation 
GeoChain
    geometry translation testing
CSG
    New geometry model
CSGOptiX
    CSG intersection with OptiX 7 
qudarap
    CUDA optical photon simulation, CUDA upload, download, textures
u4
    New Geant4 interface, genstep collection, U4Recorder 
gdxml
    GDML loaded as XML for fixups 
g4cx
    New top level package integrating Geant4 and CSGOptiX


SKIP Other packages
--------------------

ana
    python analysis tools

SKIP Old Top Level Test Packages
---------------------------------

ok
    top level Opticks, no Geant4 dependency
g4ok
    old G4Opticks interface : Opticks embedded in Geant4 
okg4
    old top level : Geant4 embedded in Opticks 


SKIP Retired Packages
-----------------------

assimprap
    old Assimp COLLADA machinery 
integration
    testing of small geometries authored in python
numpyserver
    Former expt 
openmeshrap
    Old mesh fixing tools 
analytic
    GDML python parsing  
yoctoglrap
    Old GLTF export using yoctogl


