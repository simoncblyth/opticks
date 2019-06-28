plan-removal-of-legacy-geometry-workflow-packages-and-externals
==================================================================


Currently excluding Boost, CUDA and OptiX around 16 externals.
Are transitioning to a better direct geometry approach that prevents the need 
for 6 of them, which I plan to eliminate.

Mainline externals, no plan to remove::

    plog             : logging 
    bcm              : boost CMake modules, CMake target import/export without the boilerplate 
    glm              : vector/matrix math
    oyoctogl         : GLTF handling,  nlohmann json  
                       (am using GLTF much less now, it could become optional : but need nlohmann json)

    glfw             : OpenGL
    glew             : OpenGL
    gleq             : minimal header for event handling from GLFW author
    imgui            : OpenGL interface

    xercesc          : XML parsing, needed by g4 persistency (GDML handling) 
    g4               : Geant4 (monster dependency)


To be removed::

    assimp           : used for loading COLLADA (plan to eliminate)
    openmesh         : mesh fixing (plan to eliminate, no longer used)
    opticksdata      : holds G4DAE COLLADA (plan to eliminate), moving to an "opticksgdml" to hold GDML files only (Geant4 geometry format)
    oimplicitmesher  : implicit meshing (plan to eliminate)
    odcs             : dual contouring sample, (plan to eliminate)  
    ocsgbsp          : binary space partitioning experiment (plan to eliminate)





But how to go about that. Just deleting all that work is too difficult, even though
still here in repo.

Staged removal:

1. make them optionals, remove from standard list with macros to prevent compilation of wrappers  
2. migrate them with history into another repo, perhaps named "OpticksMisc" and delete them from Opticks 


Complications
--------------

* need opticksgdml repo to take over from opticksdata
* old geometry workflow still used as OKTest etc default, need an equivalent in new way using 
  sample gdml files in opticksgdml as the input  


Reminder on Legacy Workflow 
-----------------------------

GScene is only used in legacy geometry workflow. It is intended 
for legacy workflow and GScene to be eliminated, 
but thats going to require major surgery : so for now have to live with it.

Legacy workflow has separate triangulated and analytic geometry routes, whereas
the direct workflow does these together.

Triangulated
   
    * Simultaneous G4DAE export of COLLADA DAE+GDML from Geant4 in memory geometry
    * import of G4DAE COLLADA with assimprap populating GGeo

Analytic 

    * python parsing of GDML writing to a transport GLTF format with binary .npy "extras" 
      destined to become NCSG 



To see where GScene comes in::

    GScene=ERROR OKTest --xanalytic --gltf 1
        ## GScene does very little postcache, it does its work precache 

    GScene=ERROR GGeoLib=ERROR OKTest --xanalytic --gltf 1
        ## and and the results get loaded via GGeoLib 



