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


