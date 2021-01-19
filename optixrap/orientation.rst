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
  

