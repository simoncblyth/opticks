OptiXRap Orientation : translates GGeo->OptiX
===============================================

* :doc:`../docs/orientation`

* https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/
* https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/OGeo.cc



OScene
   steering for creation of OptiX context 

OGeo
   top level construction of OptiX geometry :doc:`OGeo`

OEvent
   used for uploading gensteps and downloading hits 

OSensorLib
   buffer of sensor efficiencies and GPU angular efficiency texture  

OCtx
   experiment with a watertight wrapper around OptiX 5/6 that 
   exposes no OptiX types in its interface  


Other classes::

    OContext.hh
    OTracer.hh
    OPropertyLib.hh
    OTex.hh
    OBuffer.hh
    OTexture.hh
    OFormat.hh
    OConfig.hh
    OPropagator.hh
    OptiXUtil.hh
    OptiXTest.hh
    OSourceLib.hh
    ORng.hh
    OProg.hh
    OLaunchTest.hh
    OGeometry.hh
    OGeoStat.hh
    OFunc.hh
    OError.hh
    ODevice.hh
    OColors.hh
    OBufPair.hh
    OBufBase.hh
    OBuf.hh
    OAccel.hh




OptiX CUDA Sources
--------------------

.. toctree::

   cu/intersect_analytic.cu
   cu/material1_propagate.cu
  
   tests/OSensorLibGeoTest.cc



Thoughts about migrating to OptiX 7
-------------------------------------

optixTrace Payload Restricted to 8*32b, no more PerRayData struct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* moving from uint4 identity to a single identity int (nodeIndex or tripletIdentity) 
  that allows lookups into identity buffers 
  would just allow all of cu/PerRayData_propagate.h  to squeeze into 8 slots.

* should the identity buffers for all GMergedMesh be combined into one 
  (with offsets being done to the indices) and a global identity index used ?
 


How much in SBT and how much in buffers ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SBT is like laying down inputs to shader programs. What is the 
benefit of access from there as opposed to access from general global memory ? 

With CSG trees vary in size greatly ? 


Hmm all the geometry intersect examples getting their data from SBT : optixGetSbtDataPointer() 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


primIdx
~~~~~~~~~~~

700pdf p19

Primitives inside a build input are indexed starting from zero. This primitive
index is accessible inside the IS, AH and CH program. The application can
choose to offset this index for all primitives in a build input with no
overhead at runtime. This can be particularly useful when data for consecutive
build inputs is stored consecutively in device memory. The primitiveIndexOffset
is only used when reporting the intersection primitive.


IAS can reference multiple GAS handles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* is it advantageous to have separate IAS for each GAS ?
* or one IAS for the entire geometry ?

GAS
~~~~~

Opticks GParts is generally concatenation of multiple GParts each from single solids.
Each solid being a CSG node tree.

In OptiX < 7 used rtBuffer at geometry context with primBuffer, partBuffer, tranBuffer, ...










