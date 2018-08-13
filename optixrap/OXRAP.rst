OXRAP : OptiXRap : OptiX basis classes
==========================================

OContext
    holds the OptiX context

OScene
    `OScene::init` creates the OptiX context and populates
    it with geometry, boundary etc.. info 

    `OScene(OpticksHub* hub)` 

    Hub gives access to the `G*Lib` : twas thinking of hub as mediator to GGeo/GScene/...
    but am now aiming towards a single GGeo approach (and eliminating GScene) : so here the hub can be replaced
    by the GGeoBase interface 

    holds the `O*Lib` converting `G*Lib` geometry into OptiX geometry::

        OGeo
        OPropertyLib
        OBndLib
        OSourceLib
        OScintillatorLib

OEvent 
    GPU buffer control : only use of hub is `getOpticks()` and `getEvent()`
    contains OpticksEvent and optix::Buffer and OBuf for each of the OpticksEvent buffers
 
OPropagator
    only use of hub is getOpticks()

    `OPropagator( OpticksHub* hub, OEvent* oevt, OpticksEntry* entry)`



ORng
OTracer
OGeoStat
OFunc
OLaunchTest
OConfig



OBuf
OBufBase
OBufPair

OptiXTest
OColors
OptiXUtil
OAccel
OProg



See Also
----------

* :doc:`../opticksgeo/OKGEO`
* :doc:`../okop/OKOP`
* :doc:`../optixrap/OXRAP`
* :doc:`../thrustrap/THRAP`


