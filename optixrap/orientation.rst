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
 

Instance Identity
~~~~~~~~~~~~~~~~~~~~


optix_7_device.h::

    324 /// Returns the traversable handle for the Geometry Acceleration Structure (GAS) containing
    325 /// the current hit. May be called from IS, AH and CH.
    326 static __forceinline__ __device__ OptixTraversableHandle optixGetGASTraversableHandle();

    388 static __forceinline__ __device__ unsigned int optixGetInstanceIdFromHandle( OptixTraversableHandle handle );


TODO: test if the above two can be combined to grab instance identity of every intersect


