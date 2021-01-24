**orientation** : Opticks Codebase Orientation for Developers
==============================================================

.. contents:: Table of Contents https://simoncblyth.bitbucket.io/opticks/docs/orientation.html
   :depth: 3


Opticks Objectives
--------------------

1. replace Geant4 optical photon simulation with an equivalent GPU implementation based on NVIDIA OptiX  
2. provide automated geometry translation without approximation, Geant4 -> Opticks (GGeo) -> NVIDIA OptiX, 
3. provide a workflow that integrates the Opticks/OptiX simulation of optical photons with 
   the Geant4 simulation of all other particles

**Page Overview**

The focus of this description is on the Opticks usage of NVIDIA OptiX however necessary 
summary contextual info for developers unfamiliar with Opticks is also collected here together
with references to more extensive documentation.

**Links**

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
----------------------------------------

.. image:: //env/Documents/Geant4OpticksWorkflow/Geant4OpticksWorkflow.001.png
  :width: 1024
  :alt: Geant4-Opticks-OptiX workflow


**G4Opticks : Geant4-Opticks interface class**

Despite the classname G4Opticks is an Opticks class from g4ok sub-project
that provides a minimal interface to using embedded Opticks. It is
intended to be integrated with the Geant4 based simulation framework 
of an experiment.

* :doc:`../g4ok/orientation`


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
from nuclear reactors or from the sun or from earths mantle or from distant galaxies. 
 

Geometry translation : Geant4->Opticks(GGeo)->OptiX (green arrows)
------------------------------------------------------------------------------------------------

The green arrows in the above workflow diagram represent the translation of geometry information 
that happens at initialization.  As this translation can take minutes for large geometries
the Opticks(GGeo) geometry model is persisted to binary *.npy* files which can act as a **geocache**.  

Geometry translation is steered by *G4Opticks::translateGeometry* with *X4PhysicalVolume*
taking the leading role.

The translation entails the serialization of Geant4 C++ geometry objects for materials, surfaces 
and solid shapes into arrays and the upload of those into GPU buffers and textures. 

Structural geometry volumes are translated into the NVIDIA OptiX geometry model using a very small 
and flat heirarchy by effectively "factorizing" the structural geometry in a way that exploits 
the large degree of repetition present in typical detector geometries such as JUNO, 
with many thousands of photomultiplier tubes of various types. This "factorization" is
done with the **GInstancer** as detailed below. 


Opticks Usage of NVIDIA OptiX
--------------------------------

Direct use of OptiX is primarily in the optixrap subproject :doc:`../optixrap/orientation`
however most of the rest of Opticks is involved with the conversion of the 
Geant4 geometry into a form that can become an OptiX geometry suitable for optical photon simulation.



Familiarity with Geant4 
-------------------------

Some familiarity with the Geant4 geometry model is required to understand Opticks
as the bulk of Opticks code is concerned with the automated translation of Geant4 
geometries into Opticks(GGeo) geometries and subsequently OptiX geometries. 

Opticks provides some simple bash functions to viewing Geant4 source, eg::

   g4-   # precursor bash function 
   g4-cls G4VSolid 


* all Geant4 class names are prefixed with "G4"
* Opticks/ggeo class names mostly start with "G"
* Opticks/npy class names mostly start with "N"
* **G4Opticks** has an exceptional name, it is an Opticks/g4ok class 


**Geant4 Links**

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


 

Geant4 classes which are partially ported to CUDA/OptiX 
----------------------------------------------------------

* G4Cerenkov (only generation loop)
* G4Scintillation (only generation loop)
* G4OpRayleigh (bulk scattering)
* G4OpAbsorption (bulk absorption)
* G4OpBoundaryProcess (only a few surface types)

To quickly view the sources of any Geant4 classes use the opticks bash function g4-cls::

    g4-;g4-cls G4Cerenkov 


Primary Opticks Packages relevant to geometry
-----------------------------------------------

The below Opticks sub-projects are the most relevant 
ones for understanding the geometry translation.
The linked sub-project orientation pages highlight a few of the more important classes. 
The quoted names are common abbeviations used for the sub-projects.

extg4 "x4"  
    :doc:`../extg4/orientation`

ggeo "ggeo" 
    :doc:`../ggeo/orientation`

optixrap "oxrap"
    :doc:`../optixrap/orientation`

npy "npy"
    :doc:`../npy/orientation`

g4ok "g4ok"
    :doc:`../g4ok/orientation`

Descriptions of all ~20 packages : :doc:`misc/packages`


Highlighted Geometry classes from Geant4 and Opticks 
-----------------------------------------------------

Geometry classes can be split into three categories:

1. material and surface properties
2. solid shapes 
3. volume structure   

The below sections list the classes from Geant4 and Opticks
that you need to be familiar with in order to understand how 
Opticks translates the Geant4 geometry into an NVIDIA OptiX 
geomety suitable for optical photon simulation.


Material and Surface property classes
----------------------------------------

Geant4 material/surface property classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

G4MaterialPropertiesTable
   holds properties such as RINDEX (refractive index), ABSLENGTH (absorption length), RAYLEIGH (scattering length)
   as a function of energy

G4Material 
   name and properties table 

G4LogicalBorderSurface
   surface properties associated with the interface between two placed volumes (PV)

G4LogicalSkinSurface
   surface properties associated with all placements of an unplaced logical volume (LV)


Opticks/ggeo material/surface property classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Material and surface properties from Geant4 
are interpolated onto a common wavelength domain 
and stored within instances of the below *ggeo* classes

GMaterial GSurface
   GPropertyMap subclasses

GMaterialLib GSurfaceLib
   vectors of GMaterial and GSurface with ability to serialize into NPY arrays

GBndLib
   holds vector of int4 where the integers are indices pointing at surfaces and materials 
   inner-material/inner-surface/outer-surface/outer-material 

* :doc:`../ggeo/orientation`
* :doc:`../optixrap/orientation`


Vital role of Boundary Indices and Boundary Texture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This GGeo/GBndLib boundary lib is converted by oxrap/OBndLib into 
the GPU boundary texture interleaving properties from the GMaterialLib and GSurfaceLib 
into the boundary texture.  

During initial traversal of the Geant4 volume tree a boundary index is assigned 
to every volume corresponding to unique combinations of the 
four indices (inner-material,inner-surface,outer-surface,outer-material).

This boundary index together with other identity information from *GVolume::getIdentity*
is available GPU side in the identity buffer.

The boundary index together with the boundary texture allows the current and next 
material properties and the relevant surface properties to be obtained
via wavlength interpolated lookups on the boundary texture.
 

Geometrical shape classes 
----------------------------

G4VSolid 
   abstract base class for solids such as G4Sphere, G4Cons (cone), ...
   Translated into NCSG containing trees of nnode (npy)

npy/nnode
   constituent of CSG trees held in NCSG 

ggeo/GParts
   holder of NCSG nnode trees, with concatenation capability 

ggeo/GMesh
   despite the name this encompasses both triangulated mesh and analytic CSG geometry
   of distinct solid shapes

ggeo/GMergedMesh
   merged mesh containing merges of both triangulated and analytic geometry representations
   from multiple GMesh.  

ggeo/GGeoLib
   holds the GMergedMesh and handles persisting   


* :doc:`../npy/orientation`
* :doc:`../ggeo/orientation`


Volume Structure classes in Geant4 and Opticks/GGeo
-----------------------------------------------------


G4LogicalVolume "LV"
   unplaced volume having a solid and material

G4VPhysicalVolume "PV"
   placed volume positioning the G4LogicalVolume within a hierarchy  

ggeo/GVolume 
   converted from Geant4 physical+logical volume, has GMesh and transform constituents

ggeo/GMergedMesh
   as GMergedMesh holds identity and transform arrays across the entire geometry it straddles both 
   shape and structure geometry categories

   GMergedMesh are Translated into optix::Group OR optix::GeometryGroup underneath 
   a top level *m_top* optix::Group by optixrap/OGeo.

ggeo/GGeo
   top level geometry object holding instances of GNodeLib, GMeshLib, GBndLib, GMaterialLib, GSurfaceLib, ...

optixrap/OGeo
   converts the GGeo/GGeoLib/GMergedMesh into shallow optix::Group optix::GeometryGroup tree. 
   The approach taken was chosen because it allows instances to 
   have variables assigned as allowing instance indices to 
   be associated and thus providing identity information for all intersects.
   Details in :doc:`../optixrap/OGeo` 
    

Critical Role of ggeo/GInstancer : GVolume tree -> GMergedMesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Geant4 to GGeo conversion first recreates the full 
volume hierarchy in a tree of GVolume.  As detector geometries generally 
have large numbers of identical assemblies of multiple volumes it is 
important to fully exploit instancing to allow the geometry to fit 
into available memory.

For example the JUNO photomultipler tubes of various types are each 
modelled in geant4 with small tree of less than 10 volumes each.  These 
assemblies are then repeated many thousands of times forming the full geometry.
Other pieces of geometry such as very large acrylic spheres and support structures
are not sufficiently repeated to warrant instancing.

The ggeo/GInstancer automatically identifies repeated assemblies of volumes
using a so called progeny digest for every node of the geometry that incorporates
the shapes of the children of a node and their relative transforms.
Looking for repetitions of the progeny digest and disallowing repeats that 
are contained within other repeats allow all nodes of the geometry to 
be assigned with a repeat index (*ridx*). Remainder volumes which 
do not pass instancing criteria such as the number of repeats are assigned
repeat index zero.

The JUNO geometry contains only about 10 distinct repeated assemblies of volumes, 
including 4 or 5 different types of photomultipler tubes, various support structures
as well as the remainder miscellaneous non-repeated volumes. 
Traversals allow the global transforms of each of these repeated assemblies to be 
collected into arrays. The remainder volumes of course only have one transform : the identity matrix.

The subtree of GVolumes of the first occurrence of each repeated assembly are combined
together into GMergedMesh instances. Thus the full JUNO geometry "factorizes" into 
about 10 GMergedMesh instances with each having arrays of up to 30,000 4x4 transforms.


GMergedMesh -> optix::Group OR optix::GeometryGroup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The details of the OptiX geometry structure are documented in 

* :doc:`../optixrap/orientation` 
* :doc:`../optixrap/OGeo`

With simulation it is necessary to know the identity of the instance for every geometry intersect
as different instances can have different efficiencies and as it is expensive to reconstruct 
identity just from position.

The need to assign an index to the instances is the reason behind the choice of 
NVIDIA OptiX geometry structure.


OptiX raygen function for simulation
-----------------------------------------

optixrap/cu/generate.cu
   :doc:`../optixrap/cu/generate.cu`

   Steers photon generation and propagation from input Scintillation/Cerenkov/Torch gensteps

optixrap/cu/propagate.h
   :doc:`../optixrap/cu/propagate.h`

   Included into generate.cu, provides a CUDA port of Geant4 optical physics. 


OptiX closest hit functions
-----------------------------

optixrap/cu/material1_propagate.cu
   :doc:`../optixrap/cu/material1_propagate.cu`

   Used for photon simulation, populating PerRayData_propagate.h

optixrap/cu/material1_radiance.cu 
   :doc:`../optixrap/cu/material1_radiance.cu`

   Used for creation of ray trace images of geometry, populating PerRayData_radiance.h


OptiX ray-geometry intersect and bounds functions
-----------------------------------------------------

optixrap/cu/intersect_analytic.cu
   :doc:`../optixrap/cu/intersect_analytic.cu`

   CUDA/OptiX bounds and intersect functions for general CSG shapes created into OptiX RTprogram by::

       optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm)

optixrap/cu/csg_intersect_boolean.h
   :doc:`../optixrap/cu/csg_intersect_boolean.h`

   Evaluative CSG : iterative implementation of recursive algorithm 

optixrap/cu/csg_intersect_part.h
   :doc:`../optixrap/cu/csg_intersect_part.h`

   Large switch statements using the bounds and intersect functions from *cu/csg_intersect_primitive.h*

optixrap/cu/csg_intersect_primitive.h
   :doc:`../optixrap/cu/csg_intersect_primitive.h`

   Bounds and intersect functions for of order 10 primitive shapes::

       csg_bounds_convexpolyhedron, csg_intersect_convexpolyhedron  
       csg_bounds_cone,             csg_intersect_cone
       ...

optixrap/cu/pack.h
   :doc:`../optixrap/cu/pack.h`

   PACK4 UNPACK4_0/1/2/3 macros 


../dev/csg/index
   :doc:`../dev/csg/index`

   CSG algorithm development notes, including "slavish" python of the CSG implementation
 


CMake Structure 
-----------------

Opticks is structured as a collection of ~20 modular CMake sub-projects organized by 
their dependencies. The sub-projects are hooked together into a tree using the CMake *find_package* mechanism 
which uses *BCM* (Boost CMake Modules) to reduce CMake boilerplate.  The upshot is that 
you only need to worry about one level of dependencires 

Bash Functions
----------------

Bash functions are used for building the tree of CMake projects, see *om.bash*



