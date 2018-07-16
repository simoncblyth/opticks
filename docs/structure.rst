Structure of Opticks 
=====================

Opticks is structured into ~20 separate CMake projects 
organized by their dependencies on each other and 
on ~17 external projects.  

This structure was adopted for development speed and flexibility, 
allowing (in principal) for OpenGL visualization of geometry/propagations 
even on machines without a suitable NVIDIA GPU.

Opticks projects
------------------

OKConf
    detects versions of Geant4, OptiX available 
SysRap
    low level system utils
BoostRap
    wrapper for boost file_system, program_options etc...
NPY
    numerical workhorse sub-project

    * buffer persistency in NumPy format
    * mesh handling/polygonization
    * nnode, NCSG : CSG node tree representation of geometry   

    (currently this package is too big, it needs to be split) 

YoctoGLRap
    wrapper for the YoctoGL external, providing glTF 2.0 3D 
    file format parsing/writing  
OpticksCore
    commandline parsing, app config/steering  

GGeo
    Opticks geometry model ready for translation to GPU 
ExtG4
    translates Geant4 solids into NCSG trees and Geant4 trees of volumes
    into GGeo geometries 
AssimpRap
    wrapper for Assimp 3D geometry importer, can load G4DAE COLLADA geometry files
    (no longer needed with ExtG4 direct from Geant4 conversion)
OpenMeshRap
    wrapper for OpenMesh, providing mesh traversal : used for mesh fixing 
    (no longer needed with analytic geometry) 
OpticksGeo
    OpticksHub : non-viz hostside intersection of configuraton, geometry and event

CUDARap
    wrapper for CUDA, used for RNG seed generation/persistance 

ThrustRap
    wrapper for Thrust (higher level C++ access to CUDA), used for 

    * photon seeding(associating photon "slots" with their gensteps prior to generation)  
    * copying back the hits 

OptiXRap
    wrapper for OptiX
 
    * CUDA code for primitive and CSG intersection 
    * converts GGeo geometry into OptiX geometry, using OGeo
    * sets up OptiX context using OScene, OEvent
    * provides OptiX GPU launcher in OPropagator 
   
OGLRap
    wrapper for OpenGL, visualization of geometry and photon propagations

OKOP
    pure compute propagation, with no OpenGL dependency       

OpticksGL
    OpenGL/OptiX/CUDA interop propagation using shared OpenGL buffers, 
    allowing visualization of propagations directly from the shared interop
    GPU buffers 
     
OK
    high level manager with OpenGL visualization  

CFG4
    comparison of Opticks and Geant4 photon propagations, by 
    running both Geant4 and Opticks propagations from same executable

OKG4
    top level with visualization project intended for using 
    Opticks with Geant4 embedded inside, for example to enable 
    a Geant4 particle gun to be used within the Opticks context     
     
G4OK
    top level (non-visualization) project intended to provide simple 
    interface between Geant4 code with embedded Opticks : to be 
    used from Geant4 examples




Opticks Externals
--------------------

Following the adoption of fully analytic geometry and the
direct from Geant4 approach many of these externals are no
longer needed.  


Base externals
~~~~~~~~~~~~~~~~

bcm
    boost CMake modules, target export/import for CMake 3.5+ 
    allows config to direct dependencies only, the rest of the tree
    gets configured automatically  
boost
    system, program_options, filesystem, regex
glm
    vector, matrix, 3D projection mathematics
plog
    logging   
xercesc
    XML parsing needed by g4
g4
    geant4 


Visualization externals
~~~~~~~~~~~~~~~~~~~~~~~~~

glfw
    cross platform OpenGL and system events : keyboard, mouse  
gleq
    event queue for glfw  
glew
    OpenGL extension wrangler, providing access to OpenGL symbols 
imgui
    immediate mode OpenGL GUI     


Mesh manipulation and polygonization externals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All these are not needed with direct from G4 workflow, they 
are used for mesh manipulation and polygonization functionality.

openmesh
    provides mesh traversal 
oimplicitmesher
odcs
ocsgbsp
    polygonization
     

Other externals
~~~~~~~~~~~~~~~~~~~~

Only oyoctogl is still needed.

assimp
    used for COLLADA DAE file format loading  
    (not needed with direct from G4 workflow) 
opticksdata
    common repository for geometry 
    (not needed with direct from G4 workflow) 
oyoctogl
    glTF 2.0 3D file format parsing/construction 




Geometry translation steps
============================

Outline of the steps to translate Geant4 geometry to OptiX GPU Geometry

1. analyse geometry to find different types of repeated instances of geometry together
   with the global geometry of solids that are not repeated enough to pass the 
   instancing cuts.  Repeats that are contained within other repeats are disqualified 
   in order to end up with "assemblies" of multiple volumes. 
   This for example finds the ~5 volumes that comprise the JUNO PMTs and 
   all their 4x4 transforms. 

2. convert each G4VSolid into a Opticks nnode/NCSG tree 

3. balance the NCSG tree by:

   a) positivization : removing all subtractions in the tree by application of DeMorgans 
      rules pushing negations into complemented primitives makes the tree easier to
      rearrange as it then contains commutative unions or intersections only

   b) rearrange the tree to make more balanced

   Balancing the tree is needed as many boolean solids (eg repeated subtractions) 
   yield imbalanced trees which are inefficiently handled as complete binary trees. 

4. serialize the CSG tree of each solid into buffers

5. serialize the structure of the full geometry into buffers for each instance
   as well as for the global non-instanced geometry

6. interleave all material and surface properties as a function of wavelength 
   into a buffer ready for conversion into a GPU texture   

7. apply the NVIDIA OptiX API to put the entire geometry into GPU buffers






