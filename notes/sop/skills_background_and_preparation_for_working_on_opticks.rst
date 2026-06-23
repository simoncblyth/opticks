skills_background_and_preparation_for_working_on_opticks
=========================================================

Overview
----------

This document outlines background knowledge that will help you
to contribute to Opticks and Opticks+JUNOSW.
Clearly the depth of knowledge needed depends on what you
are working on - nevertheless some familiarity with all
areas is needed.


General Approach
-----------------

Take text file notes of what you are working on, kept in a git repository on github/gitlab

* copy/paste text of commands you are running and errors you encounter
* include questions regarding things you do not understand with answers from AI or others


Bash Scripting
----------------

Bash is the glue that links everything together,
so you need to be comfortable using bash, especially
bash functions.


General knowledge
--------------------

* Parallelized Speedup Math : Amdahls Law


General Software skills
-------------------------

* git : commit, push, pull, branch
* github, gitlab
* gitlab-ci/cd : automated software testing and distribution

* python : especially NumPy array inspection/manipulation, matplotlib, pyvista
* ipython : interactive python CLI
* Python Debugger (PDB)
* C++17
* software building : gcc, ld, nm, c++filt
* GNU Debugger (gdb)
* CUDA (nvcc)
* unit testing : ctest

* slurm : sbatch, srun, srun with tmux
* RST - reStructuredText : https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html



AI Usage
---------

* Deepseek, Gemini, Claude, ...

Discuss what you are working on with AI.
Ask AI if things can be done better or alternatively.

* Do not believe everything AI(or anyone) tells you.

Code generation with AI can be helpful, but keep it small scale and with tests.


Opticks references (3 versions, github.io is most current)
-----------------------------------------------------------

Look at some presentations and CHEP Proceedings linked from the below

* https://juno.ihep.ac.cn/~blyth/
* https://simoncblyth.github.io
* https://simoncblyth.bitbucket.io


Opticks packages
-----------------

Primary packages with some principal structs from each.

sysrap - low level types
    stree, SEvt, NP, NPFold, ...

CSG - geometry model
    CSGFoundry, CSGNode, CSGPrim, ...

QUDARap - CUDA optical physics
    QSim, qsim, QEvt, QBnd, ...

CSGOptiX - converts CSGFoundry geometry to OptiX
    CSGOptiX, PIP, SBT, GAS, IAS, ...

U4 - converts Geant4 geometry to CSGFoundry model, collects gensteps
    U4, U4Tree, U4Solid

G4CX - top level interface used from Geant4 using code
    G4CXOpticks



Opticks code (3 git repo, github is the most current)
---------------------------------------------------------

* https://code.ihep.ac.cn/blyth/opticks/
* https://github.com/simoncblyth/opticks
* https://bitbucket.org/simoncblyth/opticks/

Related *np* repository is the source of NP.hh NPFold.h headers which are copied into opticks/sysrap,
but edited only from *np* repository.

* https://github.com/simoncblyth/np



Geant4 : especially geometry optical photon generation and optical physics
----------------------------------------------------------------------------

Geant4 classes you should be familiar with are listed below. Examine and
understand the source code of the below classes.

Geometry definition using classes:

* G4VSolid : G4Box, G4Orb, G4Sphere, ...
* G4LogicalVolume
* G4VPhysicalVolume
* G4Material

Optical photon generation:

* G4Cerenkov
* G4Scintillation

Optical Physics:

* G4OpBoundaryProcess
* G4OpAbsorption
* G4OpRayleigh


Ray Tracing
-------------

* ray tracing vs rasterization
* ray tracing - intersection math : solving polynomials to find intersects
* CSG : Constructive Solid Geometry


Graphics Programming
---------------------

* transformation pipeline

  * model (M), view (V), projection (P) matrices
  * GLM : OpenGL Mathematics - https://github.com/g-truc/glm

* OpenGL shaders : vertex, geometry, fragment


NVIDIA CUDA
------------

* download and explore SDK examples
* write a test combining bash, C++ and CUDA that exercises kernel launching
* CUDA Thrust, high level C++ CUDA thrust


NVIDIA OptiX Ray Tracing Engine (v7+ API)
--------------------------------------------

* download and test your install by running SDK examples
* know how OptiX represents geometry

GAS
   geometry acceleration structure
IAS
   instance acceleration structure
SBT
   shader binding table



NVIDIA RTX
-----------

* familiarity with different GPU types


