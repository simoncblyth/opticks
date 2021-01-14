**orientation** : Opticks Codebase Orientation for Developers
==============================================================

.. contents:: Table of Contents
   :depth: 2


Page Overview
---------------

The focus of this description is on the Opticks usage of NVIDIA OptiX however necessary 
summary contextual info for developers unfamiliar with Opticks is also collected here together
with references to more extensive documentation.

Opticks Objectives
--------------------

1. replace Geant4 optical photon simulation with an equivalent GPU implementation based on NVIDIA OptiX  
2. provide automated geometry translation without approximation, Geant4 -> Opticks (GGeo) -> NVIDIA OptiX, 
3. provide a workflow that integrates the Opticks/OptiX simulation of optical photons with 
   the Geant4 simulation of all other particles


Links 
------

.. table::
    :align: center

    +----------------------------------------------+---------------------------------------------------------+
    | https://bitbucket.org/simoncblyth/opticks    | very latest code repository, unstable, breakage common  |     
    +----------------------------------------------+---------------------------------------------------------+
    | https://github.com/simoncblyth/opticks       | "releases" weeks/months behind, more stable             |     
    +----------------------------------------------+---------------------------------------------------------+
    | https://simoncblyth.bitbucket.io             | presentations and videos                                |    
    +----------------------------------------------+---------------------------------------------------------+
    | https://groups.io/g/opticks                  | forum/mailing list archive                              |    
    +----------------------------------------------+---------------------------------------------------------+
    | email:opticks+subscribe@groups.io            | subscribe to mailing list                               |    
    +----------------------------------------------+---------------------------------------------------------+ 


Geant4-Opticks-NVIDIA_OptiX workflow
--------------------------------

.. image:: //env/Documents/Geant4OpticksWorkflow/Geant4OpticksWorkflow.001.png
  :width: 1024
  :alt: Geant4-Opticks-OptiX workflow


G4Opticks : Geant4-Opticks interface class
--------------------------------------------

G4Opticks provides a minimal interface to using embedded Opticks. It is
intended to be integrated with the Geant4 based simulation framework 
of an experiment.

* https://bitbucket.org/simoncblyth/opticks/src/master/g4ok/G4Opticks.hh
* https://bitbucket.org/simoncblyth/opticks/src/master/g4ok/G4Opticks.cc

Geometry translation : Geant4->Opticks(GGeo)->OptiX (green arrows)
------------------------------------------------------------------------------------------------

The green lines in the above workflow diagram represent the translation of geometry information 
that happens at initialization.  As this translation can be take minutes for large geometries
the Opticks(GGeo) geometry model is persisted to binary *.npy* files which act as a **geocache**.  

Geometry translation is steered by *G4Opticks::translateGeometry* with *X4PhysicalVolume*
taking the leading role.


extg4 "x4"  
    translates Geant4->GGeo only

ggeo "ggeo"
    model the geometry : GMaterialLib, GSurfaceLib, GBndLib, GNodeLib, GGeoLib, ... 
    provide binary persistency in *.npy* arrays

optixrap "oxrap"
    translates GGeo->OptiX 


* https://bitbucket.org/simoncblyth/opticks/src/master/extg4/
* https://bitbucket.org/simoncblyth/opticks/src/master/extg4/X4PhysicalVolume.cc
* https://bitbucket.org/simoncblyth/opticks/src/master/ggeo/
* https://bitbucket.org/simoncblyth/opticks/src/master/ggeo/GGeo.cc
* https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/
* https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/OGeo.cc


Geant4 Links
--------------

Some familiarity with the Geant4 geometry model is required to understand Opticks
as the bulk of Opticks code is concerned with the automated translation of Geant4 
geometries into Opticks(GGeo) geometries and subsequently OptiX geometries. 

* https://geant4.web.cern.ch
* https://geant4-userdoc.web.cern.ch/UsersGuides/ForToolkitDeveloper/BackupVersions/V10.6c/html/OOAnalysisDesign/Geometry/geometry.html
* https://geant4-userdoc.web.cern.ch/UsersGuides/IntroductionToGeant4/html/index.html


Gensteps (blue arrows) and hits (red arrows)
----------------------------------------------

Gensteps (blue arrows in the above workflow diagram) are small arrays of shape `(num_gensteps,6,4)` 
which act to carry the parameters of the Cerenkov and Scintillation photon generation from 
their origin in the modified Geant4 process classes to the CUDA ports 
of the process generation loops running within the NVIDIA OptiX ray generation program. 

The parameters of each genstep includes the number of photons to generate and the 
line segment along which to generate them together with other parameters 
needed by the port of the G4Cerenkov and G4Scinillation generation loops.

Gensteps are typically several orders of magnitude smaller than the photons
that they yield. Photon generation on GPU has double benefits:

1. no copying of lots of photons from CPU to GPU
2. no CPU memory allocation for the majority of the photons, only the small 
   fraction of photons that are detected, known as *hits*, need to have CPU memory 
   allocation (see red arrows in the above workflow diagram)

Gensteps are the inputs to the optical simulation which yield hits as the output. 


OptiX GPU use of gensteps to generate photons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/cu/cerenkovstep.h
* https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/cu/scintillationstep.h
* https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/cu/generate.cu 

Geant4 CPU collection of gensteps 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://bitbucket.org/simoncblyth/opticks/src/master/g4ok/tests/G4OKTest.cc
* https://bitbucket.org/simoncblyth/opticks/src/master/examples/Geant4/CerenkovMinimal/src/L4Cerenkov.cc
* https://bitbucket.org/simoncblyth/opticks/src/master/examples/Geant4/CerenkovMinimal/src/L4Cerenkov.cc


Physics Background
--------------------

Scintillation and Cerenkov are the two physical processes which are the principal sources 
of light relevant to neutrino and dark matter experiments.

* https://en.wikipedia.org/wiki/Scintillation_(physics)
* https://en.wikipedia.org/wiki/Cherenkov_radiation

After generation the photons propagate through the detector geometry 
being scattered, absorbed, reemitted (in the bulk) and reflected, refracted, detected
or absorbed (on surfaces encountered).  

The objective of simulation is to provide estimates of times and numbers of photons 
that reach detectors such as Photomultipler tubes (PMTs) by creation of large samples with 
various input particles types and parameters. 

Simulation is the best way to understand complex detectors and as a result 
form a better understanding of the physics of interest such as neutrinos coming 
from nuclear reactors or from the sun or from distant galaxies. 
  

Geant4 classes which are partially ported to CUDA/OptiX 
----------------------------------------------------------

* G4Cerenkov (only generation loop)
* G4Scintillation (only generation loop)
* G4OpBoundaryProcess

To quickly view the sources of any Geant4 classes use the opticks bash function g4-cls::

    g4-;g4-cls G4Cerenkov 

CMake Structure 
-----------------

Opticks is structured as a collection of ~20 modular CMake sub-projects organized by
their dependencies. The sub-projects are hooked together into a tree using the CMake *find_package* mechanism 
which uses *BCM* (Boost CMake Modules) to reduce CMake boilerplate.  The upshot is that 
you only need to worry about one level of dependencires 

Bash Functions
----------------

Bash functions are used for building the tree of CMake projects, see *om.bash*

Roles of the Opticks sub-projects
-----------------------------------





